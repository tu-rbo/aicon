"""
Generating Videos from Single Image Files

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import glob
import os
import shutil
import subprocess
import cv2
from tqdm import tqdm


def find_ffmpeg():
    """Return a usable ffmpeg executable, or None if there is none."""
    exe = shutil.which("ffmpeg")
    if exe is not None:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_video_for_path(base_path, FPS=60, file_ending=".png", crf=23, preset="slow"):
    """Encode all frames in base_path into a compressed video.

    Uses H.264 (libx264) via ffmpeg when available, with quality set by crf (0 = lossless, 18 = visually lossless,
    23 = default, 28 = small). Falls back to OpenCV's MPEG-4 writer otherwise.
    """
    frame_paths = sorted(glob.glob(base_path + "/*" + file_ending))
    if not frame_paths:
        print("No " + file_ending + " frames in " + str(base_path))
        return

    ffmpeg = find_ffmpeg()
    s = None
    proc = None
    out = None

    for p in tqdm(frame_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("Could not read " + p + ", skipping")
            continue
        if s is None:
            s = img.shape
            if ffmpeg is not None:
                proc = subprocess.Popen(
                    [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-s",
                        "%dx%d" % (s[1], s[0]),
                        "-r",
                        str(FPS),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        preset,
                        "-crf",
                        str(crf),
                        # yuv420p keeps the file playable everywhere; needs even dims
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        os.path.join(base_path, "output.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                )
            else:
                out = cv2.VideoWriter(
                    os.path.join(base_path, "output.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS,
                    (s[1], s[0]),
                )
                if not out.isOpened():
                    raise RuntimeError("Could not open OpenCV mp4v writer")
        if s != img.shape:
            print("Skipping " + p + " with shape " + str(img.shape) + " != " + str(s))
            continue
        if proc is not None:
            proc.stdin.write(img.tobytes())
        else:
            out.write(img)

    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed for " + str(base_path))
    elif out is not None:
        out.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode the frames in each given directory into a video.")
    parser.add_argument("frame_dirs", nargs="+")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    for p in args.frame_dirs:
        make_video_for_path(p, FPS=args.fps)
