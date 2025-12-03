import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as ort
import cv2
from utils.EdgeTAM_image_predictor_onnx import ImagePredictor
import argparse

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        # cv2.imwrite('./mask_image.jpg', mask_image)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(
    image, 
    masks, 
    scores, 
    point_coords=None, 
    box_coords=None, 
    input_labels=None, 
    borders=True,
    save_dir="./results",        # 新增：保存目录
    base_name="mask"             # 新增：基础文件名
):
    """
    保存分割结果图像到文件，不再显示。
    
    Args:
        save_dir: 保存目录（会自动创建）
        base_name: 文件名前缀，如 "mask" → "mask_1.png"
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
            
        if box_coords is not None:
            show_box(box_coords, plt.gca())
            
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            
        plt.axis('off')
        
        # 保存图像（不再 plt.show()）
        save_path = os.path.join(save_dir, f"{base_name}_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()  # 释放内存
        print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_path", type=str, default="./examples/images/truck.jpg", help="Path to the input image.")
    argparser.add_argument("--model_path", type=str, default="./onnx_models", help="Path to the ImagePredictor model.")
    argparser.add_argument("--save_dir", type=str, default="./results", help="Directory to save the output images.")
    argparser.add_argument("--input_box", type=str, default="425,600,700,875", help="Input box coordinates as x1,y1,x2,y2")
    argparser.add_argument("--input_mask", type=str, default=None, help="Path to the input mask numpy file.")
    argparser.add_argument("--input_point_coords", type=str, default="575,750", help="Input point coordinates as x1,y1 or x1,y1:x2,y2")
    argparser.add_argument("--input_point_labels", type=str, default="0", help="Input point labels as 1 or 0 or 1:0")

    args = argparser.parse_args()

    # load image
    image = np.array(Image.open(args.image_path).convert("RGB"))

    predictor = ImagePredictor(args.model_path)

    predictor.set_image(image)

    # define input prompts
    if args.input_mask is not None:
        input_mask = np.load(args.input_mask)
    else:
        input_mask = np.zeros((1, 256, 256), dtype=np.float32)

    if args.input_box is not None:
        input_box = np.array([int(x) for x in args.input_box.split(",")])
    else:
        input_box = None

    if args.input_point_coords is not None:
        input_point_coords = np.array([[int(coord) for coord in point.split(",")] for point in args.input_point_coords.split(":")])
    else:
        input_point_coords = None

    if args.input_point_labels is not None:
        input_point_labels = np.array([int(label) for label in args.input_point_labels.split(":")])
    else:
        input_point_labels = None

    if input_box is None and input_point_coords is None:
        raise ValueError("At least one of input_box or input_point_coords must be provided.")

    #only box
    # input_box = np.array([75, 275, 1725, 850])
    # input_point_coords = None
    # input_point_labels = None

    # input_box = np.array([1375, 550, 1650, 800])
    # input_point_coords = None
    # input_point_labels = None

    #only point
    # input_box = None
    # input_point_coords = np.array([[500, 375], [1125, 625]])
    # input_point_labels = np.array([1, 1])

    # input_box = None
    # input_point_coords = np.array([[500, 375], [1125, 625]])
    # input_point_labels = np.array([1, 0])

    #point + box
    # input_box = np.array([425, 600, 700, 875])
    # input_point_coords = np.array([[575, 750]])
    # input_point_labels = np.array([0])
    # input_mask = np.load("./axmodel/logits.npy")
    # predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point_coords,
        point_labels=input_point_labels,
        box=input_box,
        mask_input=input_mask,
        multimask_output=False,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    # np.save("./results/logits_onnx.npy", logits)
    print(scores)
    # visualize results
    show_masks(
        image,
        masks,
        scores,
        point_coords=input_point_coords,
        box_coords=input_box,
        input_labels=input_point_labels,
        borders=True,
    )
