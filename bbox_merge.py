import argparse
import copy
import json
from pathlib import Path
from typing import List

import pandas as pd
from template import Template
from utils import (bbox_iou, create_html_report, draw_bboxes_on_image,
                   read_template_data)


def merge_bboxes(
    bboxes: List,
    delta_x: float = 0.4,
    delta_y: float = 0.1,
    iou_threshold: float = 0.1,
):
    """Function to merge a list of bounding boxes. We extend/enlarge the bounding boxes according to
        delta_x and delta_y. Delta_x corresponds to extension on the x-axis and delta_y corresponds to the extension
        on the y-axis. If bounding boxes overlap after extension we merge them according to a IoU threshold.

    Parameters
    ----------
    bboxes : List
        A list of bounding boxes where every bbox is defined as = [x1,y1,x2,y2].
    delta_x : float
        Margin on the x-axis.
    delta_y: float
        Margin on the y-axis.
    iou_threshold: float
        IoU threshold for merging bounding boxes.

    """

    def is_in_bbox(point, bbox):
        """Function to check if point is inside bounding box.

        Parameters
        ----------
        point : List
            2D point, defined in a list point = [x1,y1]
        bbox : List
            A bounding box, defined in a list bbox = [x2,y2,x3,y3]

        Returns
        -------
        bool
            True if point is inside bbox, false otherwise.
        """
        return (
            point[0] >= bbox[0]
            and point[0] <= bbox[2]
            and point[1] >= bbox[1]
            and point[1] <= bbox[3]
        )

    def intersect(bbox: List, bbox_: List) -> bool:
        """Function to see if two bboxes intersect.

        Parameters
        ----------
        bbox : List
            The first bounding box, defined in a list box1 = [x2,y1,x2,y2]
        bbox_ : List
            The second bounding box, defined in a list box2 = [x3,y3,x4,y4]

        Returns
        -------
        bool
            True if they intersect, False otherwise.
        """
        for i in range(int(len(bbox) / 2)):
            for j in range(int(len(bbox) / 2)):
                # Check if one of the corner of bbox inside bbox_
                if is_in_bbox([bbox[2 * i], bbox[2 * j + 1]], bbox_):
                    return True
        return False

    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None

    while True:
        nb_merge = 0
        used = []
        new_bboxes = []

        # Iterate over all bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue

                # Extend bounding boxes using delta_x on x-axis & delta_y on y-axis
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x,
                    b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x,
                    b[3] + (b[3] - b[1]) * delta_y,
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x,
                    b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x,
                    b_[3] + (b_[3] - b_[1]) * delta_y,
                ]

                # if bounding boxes intersect;
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    # Merge them according to IoU threshold
                    if bbox_iou(bmargin, b_margin) > iou_threshold:
                        tmp_bbox = [
                            min(b[0], b_[0]),
                            min(b[1], b_[1]),
                            max(b_[2], b[2]),
                            max(b[3], b_[3]),
                        ]
                        used.append(j)
                        nb_merge += 1

                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None

        # If no merge is performed, then all bboxes were already merged
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes



def merge_bboxes_and_visualize(
    sampled_templates: pd.DataFrame,
    save_path: str,
    delta_x: float,
    delta_y: float,
    iou_threshold: float,
) -> None:
    """Function to merge a list of bounding boxes and visualize the result. We visualize three images:

        1. Preview image with original boundinb boxes
        2. Rendered image with original bounding boxes
        3. Rendered image with merged bounding boxes

    3. corresponds to the final output of the merging algorithm.

    Parameters
    ----------
    bboxes : List
        A list of bounding boxes where every bbox is defined as = [x1,y1,x2,y2].
    delta_x : float
        Margin on the x-axis.
    delta_y: float
        Margin on the y-axis.
    iou_threshold: float
        IoU threshold for merging bounding boxes.

    """

    ids = sampled_templates["id"].tolist()
    all_results = []

    for template_id in ids:
        print(template_id)

        template_series = sampled_templates.loc[sampled_templates["id"] == template_id]
        state = json.loads(template_series["state"].item())
        preview_path = template_series["preview"].item()

        try:

            # Preview image
            preview = Template(state, template_id, preview_path)
            res = preview.render_and_visualize(dst_image="preview", visualize_bbox=True)
            if res:
                preview.preview_image.save(save_path + "/" + str(template_id) + "_p.png")

            # Rendered image
            rendered = Template(state, template_id, preview_path)
            res = rendered.render_and_visualize(dst_image="rendered", visualize_bbox=True)
            if res:
                rendered.rendered_image.save(save_path + "/" + str(template_id) + "_r.png")

            # Rendered image with merged bboxes
            boxes = rendered.get_bbox_list(exclude_text=True, exclude_basic=True)
            new_boxes = merge_bboxes(
                boxes,
                delta_x=delta_x,
                delta_y=delta_y,
                iou_threshold=iou_threshold,
            )

            temp_template = Template(state, template_id, preview_path)
            _ = temp_template.render_and_visualize(dst_image="rendered", visualize_bbox=False, visualize_rotated_bbox=True)
            temp_template.rendered_image.save(save_path + "/" + str(template_id) + "_r_points.png")

            temp_template_2 = Template(state, template_id, preview_path)
            _ = temp_template_2.render_and_visualize(dst_image="rendered", visualize_bbox=False, visualize_rotated_bbox=False)

            visualization = draw_bboxes_on_image(new_boxes, temp_template_2.rendered_image)
            visualization.save(save_path + "/" + str(template_id) + ".png")

            all_results.append(
                [
                    str(template_id) + "_p.png",
                    str(template_id) + "_r.png",
                    str(template_id) + ".png",
                ],
            )

        except:
            continue

    create_html_report(all_results, save_path)


def main(args):
    save_path = args.save_path
    json_path = args.json_path
    num_templates = args.num_templates
    template_id = args.template_id
    delta_x = args.delta_x
    delta_y = args.delta_y
    iou_threshold = args.iou_threshold

    # Read temples.json file
    templates = read_template_data(json_path)

    # Create save_path if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Sample a set of templates or one specific template with template_id
    if template_id == None:
        sampled_templates = templates.sample(n=num_templates, random_state=1)
    else:
        sampled_templates = templates.loc[templates["id"] == template_id]

    # Apply bbox merging & visualize result
    merge_bboxes_and_visualize(
        sampled_templates,
        save_path,
        delta_x,
        delta_y,
        iou_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_path",
        required=True,
        type=str,
        help="Path to save result",
    )

    parser.add_argument(
        "--template_id",
        default=None,
        type=str,
        help="Template ID",
    )
    parser.add_argument(
        "--json_path",
        default="./templates.json",
        type=str,
        help="Path to templates.json",
    )
    parser.add_argument(
        "--num_templates",
        default=100,
        type=int,
        help="Number of templates to sample.",
    )
    parser.add_argument(
        "--delta_x",
        default=0.4,
        type=float,
        help="Percentage of bbox extension on x-axis.",
    )
    parser.add_argument(
        "--delta_y",
        default=0.0,
        type=float,
        help="Percentage of bbox extension on y-axis.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.06,
        type=float,
        help="IoU threshold for merging bounding boxes.",
    )

    args = parser.parse_args()

    main(args)