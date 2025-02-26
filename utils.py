import copy
import json
import os
from io import BytesIO
from random import randint
from typing import List

import pandas as pd
import requests
from PIL import Image, ImageDraw, UnidentifiedImageError


def read_template_data(filename: str) -> pd.DataFrame:
    """
    Function to read template data from JSON file
    """
    with open(filename) as f:
        data = json.loads(f.read())
        print(f"Loaded data from {filename}")

    df = pd.DataFrame(data)
    print(f"{len(df)} total entries")

    df = pd.DataFrame(data)

    return df


def get_random_color() -> tuple:
    """Generate a new random color with opacity
    Returns
    -------
    tuple[r,g,b,opacity]
        A tuple with the new color
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    rand_color = (r, g, b, 127)
    return rand_color


def draw_bboxes_on_image(bboxes: List, dst_image: Image, bbox_format: str = "xyxy"):
    """Function to draw bounding boxes on a PIL image

    Parameters
    ----------
    bboxes : List
        List with bounding boxes to draw
    image : PIL.Image
        Image to draw the bounding boxes on

    Returns
    -------
    Image
        PIL Image with overlayed bounding boxes
    """

    new_image = copy.copy(dst_image)

    new_image = new_image.convert("RGB")
    draw = ImageDraw.Draw(new_image, mode="RGBA")

    for bbox in bboxes:
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b, 127)

        if bbox_format == "xyxy":
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
        elif bbox_format == "xywh":
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0]) + int(bbox[2])
            y2 = int(bbox[1]) + int(bbox[3])
        else:
            raise ValueError(f"Unsupported bbox_format: {bbox_format}")

        draw.rectangle(
            (x1,y1,x2,y2),
            fill=rand_color,
        )

    return new_image


def bbox_iou(box1: List, box2: List) -> float:
    """Function to compute intersection over union (IoU) between two bounding boxes.

    Parameters
    ----------
    box1 : List
        The first bounding box, defined in a list box1 = [x2,y1,x2,y2]
    box2 : List
        The second bounding box, defined in a list box2 = [x3,y3,x4,y4]

    Returns
    -------
    float
        IoU score
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


def cdn_image_to_pil(
    object_name: str,
    max_size=(512, 512),
    bucket="cms",
    environment="production",
):
    """Convert an image served by Kittl's CDN to PIL. If svg, converts to JPG
    with white background.

    Parameters
    ----------
    object_name : str
        The path to the image as it would be in the s3 bucket, for example:
        `elements/illustrations/1623166119043-6.svg`
    max_size : tuple[int, int] | None
        If not None, resize the image to max size, keeping the aspect ratio
    environment : EnvironmentEnum
        Sets the environment where to read the image from
    bucket : BucketEnum
        Sets the bucket where to read the image from

    Returns
    -------
    tuple[Image, str]
        A tuple with the PIL image and url
    """
    # assemble image url
    enviroment_url_map = {
        "staging": "cdn.stagingdesigner.com",
        "production": "cdn.kittl.com",
    }

    if bucket == "cms":
        base_url = f"https://{enviroment_url_map[environment]}/pr:sharp/plain"
    else:
        base_url = f"https://{enviroment_url_map[environment]}/pr:sharp/plain/api"

    f"{base_url}/{object_name}"
    converted_image_url = f"{base_url}/{object_name}@png"

    if object_name.startswith("https"):
        converted_image_url = object_name

    # download file to buffer
    try:
        out_image = Image.open(
            BytesIO(requests.get(converted_image_url, timeout=60).content),
        )
        out_image.thumbnail(max_size)

        return out_image  # , image_url
    except UnidentifiedImageError:
        raise ValueError(
            f"Couldn't find a valid image at: {converted_image_url}",
        ) from None
    except Exception as exception:
        raise ValueError(
            f"Could not open the image at: {converted_image_url} - Trace: {exception}",
        ) from None


def create_html_report(all_results: List, save_path: str):
    """Function to create HTML report with bbox merge visualization

    Parameters
    ----------
    all_results : List
        List with paths to images; we visualize three images side by side - 1) preview image with bounding boxes, 2) rendered image with original bounding bboxes, 3) rendered image with merged bboxes
    save_path : str
        Path to save the HTML report

    """
    html_text = '<html><body><br><table border="0" cellspacing="0" cellpadding="0">'
    number_of_columns = 3
    number_of_rows = len(all_results)

    for row in range(number_of_rows):
        html_text += "<tr>"
        for col in range(number_of_columns):
            row * number_of_columns + col
            base = os.path.basename(all_results[row][col])
            print(all_results[row][col])
            image_without_extension = os.path.splitext(base)[0]

            html_text += (
                f"<td style='vertical-align: top; font-size: 14px;'>"
                f"{image_without_extension}<br>"
                f"<img src='{all_results[row][col]}' width='200' height='200'></td>"
            )
        html_text += "</tr>"

    output_file_path = save_path + "/merge.html"
    with open(output_file_path, "w") as html_file:
        html_file.write(html_text)
