# ================================================================
# main_video.py - Run IDSS on MP4 Video
# ================================================================
# Processes a video file frame by frame
# Runs vein detection + IDSS on each frame
# Saves annotated output video
#
# How to run:
#   python main_video.py --video myvideo.mp4
#   python main_video.py --video myvideo.mp4 --save outputs/result.mp4
#   python main_video.py --video myvideo.mp4 --skip 5
# ================================================================

import os
import time
import argparse
import torch
import cv2
import numpy as np
from skimage.morphology import skeletonize

from config import (
    MODEL_PATH, IMAGE_H, IMAGE_W,
    THRESHOLD, MIN_SEGMENT_LEN, SPUR_PRUNE_ITERS,
    REANALYZE_EVERY
)
from utils.preprocessing import smooth_mask
from utils.skeleton      import prune_spurs, extract_graph_segments, skeleton_degree
from idss.features       import extract_idss_features
from idss.rules          import apply_knowledge_rules
from idss.ahp            import compute_ahp_weights
from idss.normalize      import normalize_features
from idss.topsis         import topsis_score
from idss.insertion      import find_insertion_point


# ── Load model ────────────────────────────────────────────────────
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    return model


# ── Run IDSS on one frame ─────────────────────────────────────────
def analyze_frame(prob_np, gray_np):
    img_np   = gray_np.astype(np.float32) / 255.0
    mask_raw = (prob_np > THRESHOLD).astype(np.uint8)
    mask_sm  = smooth_mask(mask_raw)
    dist_map = cv2.distanceTransform(mask_sm, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask_sm > 0).astype(np.uint8)
    skeleton = prune_spurs(skeleton, iters=SPUR_PRUNE_ITERS)
    segments = extract_graph_segments(skeleton, min_len_px=MIN_SEGMENT_LEN)
    deg = skeleton_degree(skeleton)
    junctions = np.column_stack(np.where((skeleton > 0) & (deg >= 3)))

    if len(segments) == 0:
        return None, None, None, None

    features          = extract_idss_features(
        segments, skeleton, dist_map, prob_np, (IMAGE_H, IMAGE_W)
    )
    accepted_features = []
    accepted_indices  = []
    rule_results      = []

    for i, feat in enumerate(features):
        acc, penalty, bonus, _ = apply_knowledge_rules(feat)
        if acc:
            accepted_features.append(feat)
            accepted_indices.append(i)
            rule_results.append({"penalty": penalty, "bonus": bonus})

    if len(accepted_features) == 0:
        return None, None, None, None, None

    weights, _   = compute_ahp_weights(verbose=False)
    normalized   = normalize_features(accepted_features)
    t_scores     = topsis_score(normalized, weights)
    final_scores = [
        min(1.0, t_scores[k] * rule_results[k]["penalty"] * rule_results[k]["bonus"])
        for k in range(len(t_scores))
    ]

    best_idx        = int(np.argmax(final_scores))
    best_feat       = accepted_features[best_idx]
    best_score      = final_scores[best_idx]
    insertion_point = find_insertion_point(best_feat["path"], dist_map)

    return best_feat, insertion_point, segments, junctions


# ── Draw results on frame ─────────────────────────────────────────
def draw_ui_overlay(frame, segments, best_feat, insertion_point, junctions):
    H, W = frame.shape[:2]
    output = frame.copy()

    scale_y = H / IMAGE_H
    scale_x = W / IMAGE_W

    # 1. All veins (Green)
    if segments is not None:
        for path in segments:
            pts = np.array([[int(x * scale_x), int(y * scale_y)] for y, x in path], np.int32)
            cv2.polylines(output, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # 2. Best segment (Yellow)
    if best_feat is not None:
        best_pts = np.array([[int(x * scale_x), int(y * scale_y)] for y, x in best_feat["path"]], np.int32)
        cv2.polylines(output, [best_pts], isClosed=False, color=(0, 255, 255), thickness=3)

        # 3. Confidence Text
        if insertion_point is not None:
            iy = int(insertion_point[0] * scale_y)
            ix = int(insertion_point[1] * scale_x)

            cv2.circle(output, (ix, iy), 5, (0, 255, 255), -1)
            conf_pct = best_feat["confidence"] * 100
            text_x = ix + 30
            text_y = iy - 20
            
            cv2.line(output, (ix, iy), (text_x, text_y), (0, 255, 255), 1)
            cv2.putText(output, "Best Segment", (text_x, text_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output, f"conf. {conf_pct:.1f}%", (text_x, text_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. Avoid markers (Red)
    if junctions is not None:
        for (y, x) in junctions:
            fy = int(y * scale_y)
            fx = int(x * scale_x)
            cv2.line(output, (fx-5, fy-5), (fx+5, fy+5), (0, 0, 255), 2)
            cv2.line(output, (fx+5, fy-5), (fx-5, fy+5), (0, 0, 255), 2)

    return output


# ================================================================
# MAIN VIDEO PROCESSING LOOP
# ================================================================
def process_video(video_path, save_path=None, skip=1):
    """
    Process an MP4 video file frame by frame.

    Parameters:
        video_path : path to input MP4 file
        save_path  : path to save output MP4 (optional)
        skip       : process every Nth frame (1 = every frame)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {orig_w}x{orig_h}")
    print(f"  FPS: {orig_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing every {skip} frame(s)")

    # Setup output writer
    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, orig_fps / skip,
                                 (orig_w, orig_h))
        print(f"  Saving to: {save_path}")

    # State - reuse last IDSS result between frames
    best_feat       = None
    insertion_point = None
    best_score      = None
    mask_sm         = None
    frame_num       = 0
    processed       = 0

    fps_timer = time.time()
    fps       = 0.0

    print("\nProcessing video...")
    print("Press Q to stop early\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Skip frames if requested
        if frame_num % skip != 0:
            continue

        processed += 1

        # Convert to grayscale and resize for model
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_r  = cv2.resize(gray, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        img_f   = gray_r.astype(np.float32) / 255.0
        tensor  = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)

        # Run model
        with torch.no_grad():
            out     = model(tensor)
            prob_np = torch.sigmoid(out[0, 0]).numpy()

        # Run IDSS every N frames
        if processed % REANALYZE_EVERY == 0:
            best_feat, insertion_point, segments, junctions = analyze_frame(prob_np, gray_r)

        # Draw results on original size frame
        output = draw_ui_overlay(frame, segments, best_feat, insertion_point, junctions)

        # Show
        cv2.imshow("IDSS - Video Analysis", output)

        # Save
        if writer:
            writer.write(output)

        # FPS
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps       = processed / elapsed
            processed = 0
            fps_timer = time.time()

        # Progress
        if frame_num % 30 == 0:
            pct = (frame_num / total_frames) * 100
            print(f"  Progress: {frame_num}/{total_frames} ({pct:.1f}%)  FPS: {fps:.1f}")

        # Key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped early by user.")
            break

    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved: {save_path}")
    cv2.destroyAllWindows()
    print("Done.")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IDSS - Run on MP4 Video"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input MP4 video")
    parser.add_argument("--save",  type=str, default=None,
                        help="Path to save output MP4 (optional)")
    parser.add_argument("--skip",  type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    args = parser.parse_args()

    process_video(
        video_path = args.video,
        save_path  = args.save,
        skip       = args.skip
    )