import os
import cv2
import numpy as np
from imwatermark import WatermarkDecoder

# paths to folders
WM_ROOT = "../../Dataset/watermarked_sources"
GROUPS = ["WM_1", "WM_2", "WM_3", "WM_4", "WM_5", "WM_6", "WM_7", "WM_8"]
METHODS = ["dwtDct", "dwtDctSvd", "rivaGan"]
BIT_LENGTHS = [16, 24, 32, 40, 48, 56, 64]
MIN_SIZE = 256
UPSCALE = 1.05

WatermarkDecoder.loadModel()   # for rivaGan


def upscale_if_needed(img):
    h, w = img.shape[:2]

    if min(h, w) >= MIN_SIZE:
        return img, False
    

    scale = MIN_SIZE / min(h, w) * UPSCALE
    new_w, new_h = int(w * scale), int(h * scale)

    return cv2.resize(img, (new_w, new_h)), True

def decode_consistency(images, method, n_bits):
    dec = WatermarkDecoder('bits', n_bits)


    bits = np.array([np.array(dec.decode(img, method), dtype=int) for img in images])
    
    maj = (bits.mean(axis=0) >= 0.5).astype(int)
    agree = (bits == maj[None, :]).mean()
    return agree, maj

def find_period(bits):
    n = len(bits)
    for p in range(2, n//2 + 1):

        if n % p == 0:
            tiled = np.tile(bits[:p], n//p)

            if (tiled == bits).mean() >= 0.95:
                return p, bits[:p].tolist()
    return None, None

def main():
    for group in GROUPS:
        folder = os.path.join(WM_ROOT, group)
        if not os.path.isdir(folder):
            continue

        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not files:
            continue

        images = [cv2.imread(f) for f in files]

        h, w = images[0].shape[:2]

        upscaled = any(upscale_if_needed(im)[1] for im in images)

        images = [upscale_if_needed(im)[0] for im in images]

        best_score, best_method, best_bits, best_msg = 0, None, None, None

        for method in METHODS:
            bit_list = [32] if method == "rivaGan" else BIT_LENGTHS

            for n_bits in bit_list:
                try:
                    score, msg = decode_consistency(images, method, n_bits)
                except:
                    continue
                if score > best_score:
                    best_score, best_method, best_bits, best_msg = score, method, n_bits, msg

        if best_score > 0.8:
            out = f"{group}: best {best_method} {best_bits}‑bit, agreement {best_score*100:.1f}%"
            period, short = find_period(best_msg)
            if period and period < best_bits:
                out += f" repeats every {period} bits: {short}"
            else:
                out += f" bits: {best_msg.tolist()}"
            if upscaled:
                out += " (upscaled)"
            print(out)
        else:
            print(f"{group}: no reliable match (best {best_score*100:.1f}%)")

if __name__ == "__main__":
    main()