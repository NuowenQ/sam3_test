"""
SAM3 segmentation testing pipeline with split predict/refine stages.

Three-step workflow:
  1. Define exemplar:  draw boxes on a reference image, saved to a JSON file
  2. Run prediction:   load exemplar + text, run model, save masks to disk
  3. Run refinement:   load saved masks, click to refine interactively

Splitting prediction and refinement avoids GPU memory explosion from running
the text/exemplar grounding pipeline and interactive refinement together.

Usage:
    # Step 1: Define exemplar (draws on ref image, saves to exemplar.json)
    python bb_test.py --define-exemplar ref_image.jpg

    # Step 2: Predict — saves masks/boxes/scores to predictions/ dir
    python bb_test.py --predict images/ --exemplar exemplar.json --text "sawfish card"

    # Step 3: Refine — loads predictions, click to refine masks interactively
    python bb_test.py --refine predictions/

    # Text-only prediction (no exemplar)
    python bb_test.py --predict images/ --text "dog"

Exemplar window controls:
    Drag:         draw positive box
    Ctrl+drag:    draw negative box
    U:            undo last box
    ENTER:        save & quit
    Q:            cancel

Refinement window controls:
    Click:        add positive refine point
    Ctrl+click:   add negative refine point
    Right-click:  delete detection (remove box + mask)
    U:            undo last refine point
    Z:            undo last deletion
    R:            reset all (refinements + deletions)
    M:            toggle mask overlay
    SPACE / D:    save & next image
    Q:            save & quit
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# ExemplarPrompts — normalized coords, serializable to JSON
# ---------------------------------------------------------------------------
class ExemplarPrompts:
    def __init__(self):
        self.boxes = []   # [(cx_norm, cy_norm, w_norm, h_norm, is_positive), ...]

    def add_box(self, cx, cy, w, h, is_positive):
        self.boxes.append((cx, cy, w, h, bool(is_positive)))

    def is_empty(self):
        return not self.boxes

    def summary(self):
        if self.boxes:
            return f"{len(self.boxes)} box(es)"
        return "empty"

    def save(self, path):
        data = {"boxes": self.boxes}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        ex = cls()
        ex.boxes = [tuple(b) for b in data.get("boxes", [])]
        return ex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mask_to_yolo_polygon(mask_np, img_h, img_w, epsilon_factor=0.001):
    """Convert a binary mask (H, W) to a YOLO normalized polygon.
    Returns list of (x_norm, y_norm) pairs, or empty list if no contour found."""
    import cv2
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Pick the largest contour
    contour = max(contours, key=cv2.contourArea)
    # Simplify to reduce point count
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    contour = contour.squeeze(1)  # (N, 2) — x, y in pixels
    # Normalize to [0, 1]
    coords = []
    for x, y in contour:
        coords.append((x / img_w, y / img_h))
    return coords


def collect_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    if os.path.isdir(path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        return sorted(set(os.path.abspath(f) for f in files))
    print(f"Error: {path} is not a valid file or directory")
    sys.exit(1)


def overlay_masks(image, masks):
    rgba = image.convert("RGBA")
    if masks is None or len(masks) == 0:
        return rgba, []
    mask_np = 255 * masks.cpu().numpy().astype(np.uint8)
    n = mask_np.shape[0]
    if n == 0:
        return rgba, []
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n, 1))
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]
    for mask, color in zip(mask_np, colors):
        m = Image.fromarray(mask)
        overlay = Image.new("RGBA", rgba.size, color + (0,))
        alpha = m.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        rgba = Image.alpha_composite(rgba, overlay)
    return rgba, colors


def _draw_output_boxes(ax, boxes, scores, colors):
    if boxes is None:
        return
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else box
        cn = [c / 255.0 for c in colors[i]]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=cn, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, f"{score:.2f}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=cn, alpha=0.7),
        )


def _draw_points(ax, points_px, labels):
    """Draw click points in pixel coords."""
    for (px, py), label in zip(points_px, labels):
        color = "lime" if label == 1 else "red"
        ax.plot(px, py, marker="*", markersize=18, color=color,
                markeredgecolor="white", markeredgewidth=1.0)


def _draw_boxes_norm(ax, boxes_norm, width, height):
    """Draw normalized exemplar boxes."""
    for (cx, cy, w, h, is_pos) in boxes_norm:
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        ec = "lime" if is_pos else "red"
        rect = patches.Rectangle(
            (x1, y1), w * width, h * height,
            linewidth=2, edgecolor=ec, facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def add_point_prompt(processor, state, px_norm, py_norm, label, device):
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
    if "language_features" not in state["backbone_out"]:
        dummy = processor.model.backbone.forward_text(["visual"], device=device)
        state["backbone_out"].update(dummy)

    point = torch.tensor([px_norm, py_norm], device=device, dtype=torch.float32).view(1, 1, 2)
    lbl = torch.tensor([label], device=device, dtype=torch.bool).view(1, 1)
    state["geometric_prompt"].append_points(point, lbl)
    return processor._forward_grounding(state)


@torch.inference_mode()
def apply_exemplar_prompts(processor, state, text_prompt, exemplar):
    if text_prompt:
        state = processor.set_text_prompt(prompt=text_prompt, state=state)
    if exemplar is None or exemplar.is_empty():
        return state
    for cx, cy, w, h, is_positive in exemplar.boxes:
        state = processor.add_geometric_prompt(
            state=state, box=[cx, cy, w, h], label=is_positive
        )
    return state


def auto_find_checkpoint():
    try:
        import sam3 as _sam3
        sam3_root = os.path.join(os.path.dirname(_sam3.__file__), "..")
        candidates = [
            os.path.join(sam3_root, "..", "sam3.pt"),
            os.path.join(sam3_root, "sam3.pt"),
            os.path.expanduser("~/sam3.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# MODE 1: Define exemplar — draw on reference image, save to JSON
# ---------------------------------------------------------------------------
def define_exemplar(ref_image_path: str, output_path: str):
    """Define exemplar by drawing bounding boxes on a reference image."""
    ref_image = Image.open(ref_image_path).convert("RGB")
    width, height = ref_image.size
    ref_name = os.path.basename(ref_image_path)

    exemplar = ExemplarPrompts()
    drawn_boxes = []    # (cx, cy, w, h, is_pos) normalized

    drag = {"active": False, "start": None, "rect": None, "ctrl": False}
    result = {"action": "confirm"}

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    fig.patch.set_facecolor("#2a2a2a")
    ax.set_facecolor("#2a2a2a")

    def redraw():
        ax.clear()
        ax.axis("off")
        ax.imshow(ref_image)
        _draw_boxes_norm(ax, drawn_boxes, width, height)

        n_boxes = len(drawn_boxes)
        info_str = f"{n_boxes} box{'es' if n_boxes != 1 else ''}" if n_boxes else "no boxes yet"

        hud = (f"DEFINE EXEMPLAR: {ref_name}  [{info_str}]    "
               f"[drag] +box  [Ctrl+drag] -box  [U] undo  "
               f"[ENTER] save  [Q] cancel")
        ax.set_title(hud, color="#ffcc00", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.xdata is None or event.button != 1:
            return
        drag["start"] = (event.xdata, event.ydata)
        drag["active"] = False
        drag["ctrl"] = event.key == "control"

    def on_motion(event):
        if drag["start"] is None or event.inaxes != ax or event.xdata is None:
            return
        sx, sy = drag["start"]
        if abs(event.xdata - sx) > 5 or abs(event.ydata - sy) > 5:
            drag["active"] = True
            if drag["rect"] is not None:
                drag["rect"].remove()
            x0 = min(sx, event.xdata)
            y0 = min(sy, event.ydata)
            ec = "red" if drag["ctrl"] else "yellow"
            drag["rect"] = patches.Rectangle(
                (x0, y0), abs(event.xdata - sx), abs(event.ydata - sy),
                linewidth=2, edgecolor=ec, facecolor="none", linestyle="--",
            )
            ax.add_patch(drag["rect"])
            fig.canvas.draw_idle()

    def on_release(event):
        if event.inaxes != ax or event.xdata is None or drag["start"] is None:
            drag["start"] = None
            drag["active"] = False
            if drag["rect"] is not None:
                drag["rect"].remove()
                drag["rect"] = None
            return

        if not drag["active"]:
            # Simple click — ignore (exemplar is box-only)
            drag["start"] = None
            return

        is_positive = not drag["ctrl"]
        sx, sy = drag["start"]
        ex, ey = event.xdata, event.ydata
        x1, x2 = sorted([sx, ex])
        y1, y2 = sorted([sy, ey])
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        exemplar.add_box(cx, cy, w, h, is_positive)
        drawn_boxes.append((cx, cy, w, h, is_positive))
        tag = "positive" if is_positive else "negative"
        print(f"  {tag} box: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

        drag["start"] = None
        drag["active"] = False
        if drag["rect"] is not None:
            drag["rect"].remove()
            drag["rect"] = None
        redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            result["action"] = "confirm"
            plt.close(fig)
        elif event.key == "q":
            result["action"] = "quit"
            plt.close(fig)
        elif event.key == "u":
            if not drawn_boxes:
                return
            drawn_boxes.pop()
            exemplar.boxes.pop()
            print("  undone last box")
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()

    if result["action"] == "quit" or exemplar.is_empty():
        print("Cancelled — no exemplar saved.")
        return

    exemplar.save(output_path)
    print(f"\nExemplar saved to: {output_path}")
    print(f"  {exemplar.summary()}")
    print(f"\nNow run segmentation with:")
    print(f'  python bb_test.py --predict <images> --exemplar {output_path} --text "your prompt"')


# ---------------------------------------------------------------------------
# MODE 2: Predict — run grounding, save masks/boxes/scores to disk
# ---------------------------------------------------------------------------
def run_predict(image_paths, text_prompt, exemplar, device, threshold,
                checkpoint, output_dir, resolution=1008):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    build_kwargs = {}
    if checkpoint:
        print(f"Using checkpoint: {checkpoint}")
        build_kwargs["checkpoint_path"] = checkpoint
        build_kwargs["load_from_HF"] = False

    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold,
                              resolution=resolution)
    print(f"Model loaded (resolution={resolution}).\n")

    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        print(f"[{idx + 1}/{len(image_paths)}] {fname}")

        if device == "cuda":
            torch.cuda.empty_cache()
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)

        with torch.inference_mode():
            state = apply_exemplar_prompts(processor, state, text_prompt,
                                           exemplar)

        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is not None:
            masks = masks.squeeze(1)

        n = len(masks) if masks is not None else 0
        print(f"  => {n} object(s) detected")

        entry = {"image_path": os.path.abspath(img_path), "stem": stem}
        # Move results to CPU and free GPU memory before saving
        if masks is not None:
            masks = masks.cpu().float()
        if boxes is not None:
            boxes = boxes.cpu().float()
        if scores is not None:
            scores = scores.cpu().float()
        # Free all GPU tensors from this image's state
        del state
        if device == "cuda":
            torch.cuda.empty_cache()

        if masks is not None and n > 0:
            masks_path = os.path.join(output_dir, f"{stem}_masks.npy")
            boxes_path = os.path.join(output_dir, f"{stem}_boxes.npy")
            scores_path = os.path.join(output_dir, f"{stem}_scores.npy")
            np.save(masks_path, masks.numpy())
            np.save(boxes_path, boxes.numpy())
            np.save(scores_path, scores.numpy())
            entry["masks"] = f"{stem}_masks.npy"
            entry["boxes"] = f"{stem}_boxes.npy"
            entry["scores"] = f"{stem}_scores.npy"

            # Save YOLO segmentation format (.txt)
            img_h, img_w = image.size[1], image.size[0]
            masks_np = masks.numpy()
            txt_path = os.path.join(output_dir, f"{stem}.txt")
            with open(txt_path, "w") as f:
                for mask_i in masks_np:
                    polygon = mask_to_yolo_polygon(mask_i, img_h, img_w)
                    if not polygon:
                        continue
                    coords_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in polygon)
                    f.write(f"0 {coords_str}\n")
            entry["yolo_txt"] = f"{stem}.txt"
        manifest.append(entry)

        # Write manifest after each image so it's available even if we crash
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nPredictions saved to: {output_dir}/")
    print(f"  {len(manifest)} image(s), manifest: {manifest_path}")
    print(f"\nNow run interactive refinement with:")
    print(f"  python bb_test.py --refine {output_dir}/")


# ---------------------------------------------------------------------------
# MODE 3: Refine — load saved predictions, interactive click refinement
# ---------------------------------------------------------------------------
def run_refine(predictions_dir, device, checkpoint):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Scan for prediction files directly (no manifest needed)
    mask_files = sorted(glob.glob(os.path.join(predictions_dir, "*_masks.npy")))
    if not mask_files:
        print(f"Error: no *_masks.npy files found in {predictions_dir}")
        sys.exit(1)

    # Build entries from the npy files
    manifest = []
    for mf in mask_files:
        stem = os.path.basename(mf).replace("_masks.npy", "")
        # Find matching image in the images_test dir or from the npy's parent
        img_candidates = glob.glob(os.path.join(predictions_dir, "..", "**", f"{stem}.*"), recursive=True)
        img_path = None
        for c in img_candidates:
            if os.path.splitext(c)[1].lower() in IMAGE_EXTENSIONS:
                img_path = os.path.abspath(c)
                break
        if img_path is None:
            # Try common image dirs
            for d in [os.path.join(predictions_dir, ".."), "/home/nuowen/Projects_2026/Sam3/images_test"]:
                for ext in [".jpg", ".jpeg", ".png"]:
                    p = os.path.join(d, f"{stem}{ext}")
                    if os.path.isfile(p):
                        img_path = os.path.abspath(p)
                        break
                if img_path:
                    break
        manifest.append({"stem": stem, "image_path": img_path,
                         "masks": f"{stem}_masks.npy", "boxes": f"{stem}_boxes.npy",
                         "scores": f"{stem}_scores.npy"})

    if not manifest:
        print("No predictions found.")
        return

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    build_kwargs = {}
    if checkpoint:
        print(f"Using checkpoint: {checkpoint}")
        build_kwargs["checkpoint_path"] = checkpoint
        build_kwargs["load_from_HF"] = False

    model = build_sam3_image_model(enable_inst_interactivity=True, **build_kwargs)
    processor = Sam3Processor(model, device=device)
    print("Model loaded (interactive refinement mode).\n")

    show_masks = True

    for idx, entry in enumerate(manifest):
        img_path = entry["image_path"]
        fname = os.path.basename(img_path)
        print(f"[{idx + 1}/{len(manifest)}] {fname}")

        if not os.path.isfile(img_path):
            print(f"  Warning: image not found at {img_path}, skipping")
            continue

        image = Image.open(img_path).convert("RGB")

        # Only run backbone (set_image) — no text/grounding needed
        state = processor.set_image(image)

        # Load saved prediction results
        masks = None
        boxes = None
        scores = None
        if "masks" in entry:
            masks = torch.from_numpy(
                np.load(os.path.join(predictions_dir, entry["masks"]))
            ).to(device)
            boxes = torch.from_numpy(
                np.load(os.path.join(predictions_dir, entry["boxes"]))
            ).to(device)
            scores = torch.from_numpy(
                np.load(os.path.join(predictions_dir, entry["scores"]))
            ).to(device)

        state["masks"] = masks
        state["boxes"] = boxes
        state["scores"] = scores

        n = len(masks) if masks is not None else 0
        print(f"  => {n} object(s) from saved predictions")

        title = f"[{idx + 1}/{len(manifest)}] {fname}"
        action, show_masks, final_masks, final_boxes, final_scores = result_viewer(
            image, model, state, title, show_masks,
        )

        # Save refined results
        stem = entry["stem"]
        img_h, img_w = image.size[1], image.size[0]
        if final_masks is not None and len(final_masks) > 0:
            masks_np = final_masks.cpu().float().numpy()
            boxes_np = final_boxes.cpu().float().numpy()
            scores_np = final_scores.cpu().float().numpy()
            np.save(os.path.join(predictions_dir, f"{stem}_masks.npy"), masks_np)
            np.save(os.path.join(predictions_dir, f"{stem}_boxes.npy"), boxes_np)
            np.save(os.path.join(predictions_dir, f"{stem}_scores.npy"), scores_np)

            txt_path = os.path.join(predictions_dir, f"{stem}.txt")
            with open(txt_path, "w") as f:
                for mask_i in masks_np:
                    polygon = mask_to_yolo_polygon(mask_i, img_h, img_w)
                    if not polygon:
                        continue
                    coords_str = " ".join(f"{x:.5f} {y:.5f}" for x, y in polygon)
                    f.write(f"0 {coords_str}\n")
            print(f"  Saved refined results ({len(masks_np)} obj)")
        else:
            # All detections deleted — write empty files
            txt_path = os.path.join(predictions_dir, f"{stem}.txt")
            with open(txt_path, "w") as f:
                pass
            print(f"  Saved (0 obj)")

        if action == "quit":
            print("Quit.")
            break

    print("Done.")


def result_viewer(image, model, state, title, show_masks):
    """Show grounding results. User can click on a detected object to refine
    its mask using SAM's interactive instance predictor (predict_inst).

    Click = include point, Ctrl+click = exclude point.
    Refinement is done per-object: click inside a detected box to refine that mask."""

    # Extract grounding results (these stay fixed)
    grounding_masks = state.get("masks")
    grounding_boxes = state.get("boxes")
    grounding_scores = state.get("scores")
    if grounding_masks is not None:
        grounding_masks = grounding_masks.squeeze(1)

    n_objects = len(grounding_masks) if grounding_masks is not None else 0

    # Per-object refined masks (start as copies of grounding masks)
    # refined_masks[i] is an np.ndarray (H, W) or None (use grounding)
    refined = {
        "masks": [None] * n_objects,  # None = use grounding mask
        "logits": [None] * n_objects,  # low-res logits for iterative refinement
        "points": [[] for _ in range(n_objects)],  # [(px,py), ...]
        "labels": [[] for _ in range(n_objects)],  # [1 or 0, ...]
    }
    deleted = set()  # indices of deleted detections
    deleted_history = []  # stack for undo-delete via [Z]

    # User-created detections (clicking outside any existing box)
    custom = {
        "masks": [],      # list of np.ndarray (H, W)
        "boxes": [],      # list of np.ndarray (4,) xyxy
        "scores": [],     # list of float
        "logits": [],     # list of np.ndarray (256, 256)
        "points": [],     # list of [(px, py), ...]
        "labels": [],     # list of [1 or 0, ...]
    }

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    action = {"value": "next"}

    def get_visible_indices():
        """Return list of grounding object indices that are not deleted."""
        return [i for i in range(n_objects) if i not in deleted]

    def get_display_data():
        """Build combined masks/boxes/scores from grounding + custom, excluding deleted."""
        device = grounding_masks.device if grounding_masks is not None else "cpu"
        masks = []
        boxes = []
        scores = []

        # Grounding detections (not deleted)
        visible = get_visible_indices()
        for i in visible:
            if refined["masks"][i] is not None:
                m = torch.from_numpy(refined["masks"][i]).to(device)
                masks.append(m)
                # Recompute box from refined mask
                ys, xs = torch.where(m > 0.5)
                if len(ys) > 0:
                    box = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()],
                                       dtype=torch.float32, device=device)
                    boxes.append(box)
                else:
                    boxes.append(grounding_boxes[i])
            else:
                masks.append(grounding_masks[i])
                boxes.append(grounding_boxes[i])
            scores.append(grounding_scores[i])

        # Custom (user-created) detections
        for i in range(len(custom["masks"])):
            masks.append(torch.from_numpy(custom["masks"][i]).to(device))
            boxes.append(torch.from_numpy(custom["boxes"][i]).to(device))
            scores.append(torch.tensor(custom["scores"][i]).to(device))

        if not masks:
            return None, None, None, visible
        return torch.stack(masks), torch.stack(boxes), torch.stack(scores), visible

    def redraw():
        ax.clear()
        ax.axis("off")
        masks, boxes, scores, visible = get_display_data()
        composited, colors = overlay_masks(image, masks)
        if show_masks and masks is not None and len(masks) > 0:
            ax.imshow(composited)
        else:
            ax.imshow(image)
        if boxes is not None and scores is not None:
            _draw_output_boxes(ax, boxes, scores, colors)

        # Draw refinement points for visible grounding objects
        for i in visible:
            _draw_points(ax, refined["points"][i], refined["labels"][i])
        # Draw points for custom detections
        for i in range(len(custom["points"])):
            _draw_points(ax, custom["points"][i], custom["labels"][i])

        n_total = len(visible) + len(custom["masks"])
        n_del = len(deleted)
        del_str = f" ({n_del} deleted)" if n_del else ""
        hud = (f"{title}  —  {n_total} obj{del_str}    "
               f"[click] include  [Ctrl+click] exclude  [right-click] delete  "
               f"[U] undo pt  [Z] undo del  [R] reset all  [M] mask  [SPACE] save & next  [Q] save & quit")
        ax.set_title(hud, color="white", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def find_object_at(px, py):
        """Find which object's box contains (px, py).
        Returns ("grounding", idx), ("custom", idx), or (None, -1)."""
        if grounding_boxes is not None:
            for i, box in enumerate(grounding_boxes):
                if i in deleted:
                    continue
                x1, y1, x2, y2 = box.tolist()
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return "grounding", i
        for i, box in enumerate(custom["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            if x1 <= px <= x2 and y1 <= py <= y2:
                return "custom", i
        return None, -1

    def refine_object(obj_idx):
        """Re-run predict_inst for a grounding object with its accumulated points."""
        box = grounding_boxes[obj_idx].cpu().numpy()  # xyxy
        pts = np.array(refined["points"][obj_idx])  # (N, 2)
        lbls = np.array(refined["labels"][obj_idx])  # (N,)

        kwargs = {
            "point_coords": pts,
            "point_labels": lbls,
            "box": box[None, :],  # (1, 4)
            "multimask_output": False,
        }
        if refined["logits"][obj_idx] is not None:
            kwargs["mask_input"] = refined["logits"][obj_idx][None, :, :]

        with torch.inference_mode():
            masks_np, scores_np, logits_np = model.predict_inst(state, **kwargs)

        refined["masks"][obj_idx] = masks_np[0]
        refined["logits"][obj_idx] = logits_np[0]

    def run_custom_predict(cidx):
        """Run predict_inst for a custom (user-created) detection."""
        pts = np.array(custom["points"][cidx])
        lbls = np.array(custom["labels"][cidx])

        kwargs = {
            "point_coords": pts,
            "point_labels": lbls,
            "multimask_output": len(pts) == 1,  # multi for first click, single after
        }
        if custom["logits"][cidx] is not None:
            kwargs["mask_input"] = custom["logits"][cidx][None, :, :]
            kwargs["multimask_output"] = False

        with torch.inference_mode():
            masks_np, scores_np, logits_np = model.predict_inst(state, **kwargs)

        # Pick best mask if multimask
        best = int(scores_np.argmax())
        custom["masks"][cidx] = masks_np[best]
        custom["scores"][cidx] = float(scores_np[best])
        custom["logits"][cidx] = logits_np[best]

        # Derive bounding box from mask
        ys, xs = np.where(masks_np[best])
        if len(xs) > 0:
            custom["boxes"][cidx] = np.array([xs.min(), ys.min(), xs.max(), ys.max()],
                                              dtype=np.float32)
        else:
            custom["boxes"][cidx] = np.array([0, 0, 0, 0], dtype=np.float32)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        px, py = event.xdata, event.ydata

        # Right-click: delete the detection under cursor
        if event.button == 3:
            source, idx = find_object_at(px, py)
            if source == "grounding":
                deleted.add(idx)
                deleted_history.append(("grounding", idx))
                print(f"  Deleted grounding object {idx}")
            elif source == "custom":
                # Remove custom detection entirely
                for key in custom:
                    custom[key].pop(idx)
                deleted_history.append(("custom_removed", None))
                print(f"  Deleted custom object {idx}")
            else:
                print(f"  Right-click at ({px:.0f}, {py:.0f}) — not inside any box")
                return
            redraw()
            return

        if event.button != 1:
            return

        # Left-click: refine existing or create new detection
        is_positive = event.key != "control"
        label = 1 if is_positive else 0
        tag = "include" if label else "exclude"

        source, idx = find_object_at(px, py)

        if source == "grounding":
            # Refine existing grounding detection
            refined["points"][idx].append((px, py))
            refined["labels"][idx].append(label)
            print(f"  {tag} point on grounding obj {idx}: ({px:.0f}, {py:.0f})")
            refine_object(idx)
        elif source == "custom":
            # Refine existing custom detection
            custom["points"][idx].append((px, py))
            custom["labels"][idx].append(label)
            print(f"  {tag} point on custom obj {idx}: ({px:.0f}, {py:.0f})")
            run_custom_predict(idx)
        else:
            # Click outside any box → create new custom detection
            custom["masks"].append(None)
            custom["boxes"].append(np.array([0, 0, 0, 0], dtype=np.float32))
            custom["scores"].append(0.0)
            custom["logits"].append(None)
            custom["points"].append([(px, py)])
            custom["labels"].append([label])
            cidx = len(custom["masks"]) - 1
            print(f"  New detection from click at ({px:.0f}, {py:.0f})")
            run_custom_predict(cidx)

        redraw()

    def on_key(event):
        nonlocal show_masks

        if event.key in (" ", "d"):
            action["value"] = "next"
            plt.close(fig)
        elif event.key == "q":
            action["value"] = "quit"
            plt.close(fig)
        elif event.key == "m":
            show_masks = not show_masks
            redraw()
        elif event.key == "u":
            # Undo last point: check custom detections first (most recent),
            # then grounding refinements
            undone = False

            # Check custom detections (last one with points)
            for ci in range(len(custom["points"]) - 1, -1, -1):
                if custom["points"][ci]:
                    custom["points"][ci].pop()
                    custom["labels"][ci].pop()
                    if custom["points"][ci]:
                        run_custom_predict(ci)
                    else:
                        # No points left — remove this custom detection
                        for key in custom:
                            custom[key].pop(ci)
                    print(f"  Undone last point on custom obj {ci}")
                    undone = True
                    break

            if not undone:
                # Check grounding refinements
                last_obj = -1
                for i in range(n_objects):
                    if refined["points"][i]:
                        last_obj = i
                if last_obj < 0:
                    return
                refined["points"][last_obj].pop()
                refined["labels"][last_obj].pop()
                if refined["points"][last_obj]:
                    refine_object(last_obj)
                else:
                    refined["masks"][last_obj] = None
                    refined["logits"][last_obj] = None
                print(f"  Undone last refine point on grounding obj {last_obj}")

            redraw()
        elif event.key == "z":
            if deleted_history:
                kind, idx = deleted_history.pop()
                if kind == "grounding":
                    deleted.discard(idx)
                    print(f"  Restored grounding object {idx}")
                elif kind == "custom_removed":
                    print("  Cannot undo custom deletion (re-click to recreate)")
                redraw()
            else:
                print("  Nothing to restore")
        elif event.key == "r":
            for i in range(n_objects):
                refined["masks"][i] = None
                refined["logits"][i] = None
                refined["points"][i].clear()
                refined["labels"][i].clear()
            deleted.clear()
            deleted_history.clear()
            for key in custom:
                custom[key].clear()
            print("  All refinements, deletions & custom detections reset")
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()

    # Collect final state for saving
    final_masks, final_boxes, final_scores, _ = get_display_data()
    return action["value"], show_masks, final_masks, final_boxes, final_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SAM3 segmentation with split predict/refine stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Step 1: Define exemplar
  %(prog)s --define-exemplar ref.jpg -o my_exemplar.json

  # Step 2: Predict (saves masks to disk, then frees GPU)
  %(prog)s --predict images/ --exemplar my_exemplar.json --text "sawfish card"

  # Step 3: Refine interactively (loads saved masks)
  %(prog)s --refine predictions/

  # Text-only prediction
  %(prog)s --predict images/ --text "dog"
""",
    )
    # Mode 1: define exemplar
    parser.add_argument("--define-exemplar", metavar="IMAGE",
                        help="Draw exemplar prompts on this image and save to JSON")
    parser.add_argument("-o", "--output", default="exemplar.json",
                        help="Output path for exemplar JSON (default: exemplar.json)")

    # Mode 2: predict
    parser.add_argument("--predict", metavar="PATH",
                        help="Run prediction on image file/dir, save results to disk")
    parser.add_argument("--predictions-dir", default="predictions",
                        help="Output directory for predictions (default: predictions/)")
    parser.add_argument("--text", default="", help="Text prompt")
    parser.add_argument("--exemplar", default=None, metavar="JSON",
                        help="Path to saved exemplar JSON file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold")

    # Mode 3: refine
    parser.add_argument("--refine", metavar="DIR",
                        help="Load predictions from dir and run interactive refinement")

    # Shared
    parser.add_argument("--checkpoint", default=None, metavar="PATH",
                        help="Path to SAM3 checkpoint (.pt). Auto-detected if not set.")
    parser.add_argument("--resolution", type=int, default=1008,
                        help="Input resolution for SAM3 (default: 1008). "
                             "Lower values (e.g. 512, 640) use less GPU memory.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.checkpoint or auto_find_checkpoint()

    # ---- Mode 1: Define exemplar ----
    if args.define_exemplar:
        if not os.path.isfile(args.define_exemplar):
            print(f"Error: {args.define_exemplar} not found")
            sys.exit(1)
        print(f"Define exemplar on: {args.define_exemplar}")
        print("Draw box(es) around the object to segment, press ENTER to save, Q to cancel.\n")
        define_exemplar(args.define_exemplar, args.output)
        return

    # ---- Mode 2: Predict ----
    if args.predict:
        images = collect_images(args.predict)
        if not images:
            print("No images found.")
            sys.exit(1)
        print(f"Found {len(images)} image(s)")

        exemplar = ExemplarPrompts()
        if args.exemplar:
            if not os.path.isfile(args.exemplar):
                print(f"Error: {args.exemplar} not found")
                sys.exit(1)
            exemplar = ExemplarPrompts.load(args.exemplar)
            print(f"Loaded exemplar: {exemplar.summary()}")

        if not args.text and exemplar.is_empty():
            print("Warning: no --text and no --exemplar given. Results may be empty.")

        print(f"Loading SAM3 model on {device}...")
        run_predict(images, args.text, exemplar, device, args.threshold, ckpt,
                    args.predictions_dir, resolution=args.resolution)
        return

    # ---- Mode 3: Refine ----
    if args.refine:
        if not os.path.isdir(args.refine):
            print(f"Error: {args.refine} is not a directory")
            sys.exit(1)
        print(f"Loading SAM3 model on {device} (refinement mode)...")
        run_refine(args.refine, device, ckpt)
        return

    parser.error("Use --predict, --refine, or --define-exemplar")


if __name__ == "__main__":
    main()
