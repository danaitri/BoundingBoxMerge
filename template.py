import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from utils import cdn_image_to_pil, get_random_color


class Template:
    """
    Represents a template and provides utilities for loading,
    processing, and analyzing template information.

    Methods:
        Visualization Utilities
            - load_image_data(): Loads image data for the template.
            - load_preview_image(): Loads the preview image associated with the template.
            - draw_on_preview_image(): Draws bounding boxes (bboxes) on the preview image.
            - draw_on_rendered_image(): Draws bounding boxes (bboxes) on the preview image.
        Bounding box scaling/norm
            - scale_bboxes(): Scales bboxes
        Get functions
            - get_bbox_list(): Get a list with all the bounding boxes associated with the template.
        Preprocessing
            - apply_mask_to_image(): Apply mask to image.
            - render_template(): Renders the template considering scaling, flipping, and rotation of visual elements.
    Usage:
        template = Template(state, template_id, template_series["preview"].item())
        template.load_image_data()
        template.render_template()
        template.draw_on_rendered_image()
        template.rendered_image.save("./somewhere_in_my_disk/image.png")

    """

    def __init__(self, state, id, preview_path):
        if "boards" in state:
            self.canvas_width = int(state["boards"][0]["artboard"]["width"])
            self.canvas_height = int(state["boards"][0]["artboard"]["height"])
            self.df_objs = pd.DataFrame.from_dict(state["boards"][0]["objects"])
        else:
            self.canvas_width = int(state["artboardOptions"]["width"])
            self.canvas_height = int(state["artboardOptions"]["height"])
            self.df_objs = pd.DataFrame.from_dict(state["objects"])

        self.df_points = pd.DataFrame(index=self.df_objs.index)

        self.id = id
        self.preview_path = preview_path

        # Draw colors for visualizations
        self.draw_colors = {
            "illustration": (255, 0, 255, 127),
            "illustrationImage": (0, 0, 255, 127),
            "basicShape": (0, 255, 255, 127),
            "pathText": (255, 255, 0, 127),
            "mask": (0, 50, 255, 127),
        }

        self.images = []
        self.is_invalid = False

    # ------------------------------- Visualization -----------------------------------------------------------------------
    def load_image_data(self) -> bool:
        """
        Load image data for template
        Returns True if all images were loaded correctly, False otherwise
        """
        self.preview_image = cdn_image_to_pil(
            object_name=self.preview_path,
            environment="production",
            bucket="api",
        )

        if len(self.df_objs) == 0:
            self.images = [None]
            self.is_invalid = True

        else:
            self.image_paths = [str(x) for x in self.df_objs["objectName"].tolist()]

            self.images.extend(
                [
                    cdn_image_to_pil(path) if path != "nan" else "nan"
                    for path in self.image_paths
                ],
            )

        return any(item is None for item in self.images)

    def load_preview_image(self) -> None:
        """
        Function to load preview image
        """
        self.preview_image = cdn_image_to_pil(
            object_name=self.preview_path,
            environment="production",
            bucket="api",
        )

    def draw_on_preview_image(self) -> None:
        """
        Function to draw bboxes on preview image
        """

        self.preview_image = self.preview_image.resize(
            (self.canvas_width, self.canvas_height),
        )

        self.preview_image = self.preview_image.convert("RGB")
        draw = ImageDraw.Draw(self.preview_image, mode="RGBA")

        # self.scale_bboxes()

        for i in range(len(self.df_objs)):
            scaled_x1 = self.df_objs["left"].take([i]).item()
            scaled_y1 = self.df_objs["top"].take([i]).item()
            scaled_x2 = scaled_x1 + self.df_objs["width"].take([i]).item()
            scaled_y2 = scaled_y1 + self.df_objs["height"].take([i]).item()
            type = self.df_objs["type"].take([i]).item()
            draw.rectangle(
                (int(scaled_x1), int(scaled_y1), int(scaled_x2), int(scaled_y2)),
                outline=self.draw_colors[type],
                width=5,
            )

    def draw_on_rendered_image(self) -> None:
        """
        Function to draw bboxes on rendered image
        """

        self.rendered_image = self.rendered_image.resize(
            (self.canvas_width, self.canvas_height),
        )

        self.rendered_image = self.rendered_image.convert("RGB")
        draw = ImageDraw.Draw(self.rendered_image, mode="RGBA")

        # self.scale_bboxes()

        for i in range(len(self.df_objs)):
            rand_color = get_random_color()

            type = self.df_objs["type"].take([i]).item()

            if type == "pathText":
                continue

            scaled_x1 = self.df_objs["left"].take([i]).item()
            scaled_y1 = self.df_objs["top"].take([i]).item()
            scaled_x2 = scaled_x1 + self.df_objs["width"].take([i]).item()
            scaled_y2 = scaled_y1 + self.df_objs["height"].take([i]).item()

            draw.rectangle(
                (int(scaled_x1), int(scaled_y1), int(scaled_x2), int(scaled_y2)),
                fill=rand_color,
            )

    def visualize_rotated_bbox(self) -> None:
        """
        Function to visualize rotated bboxes on rendered image
        """

        self.rendered_image = self.rendered_image.resize(
            (self.canvas_width, self.canvas_height),
        )

        self.rendered_image = self.rendered_image.convert("RGB")
        draw = ImageDraw.Draw(self.rendered_image, mode="RGBA")

        # self.scale_bboxes()

        for i in range(len(self.df_objs)):
            rand_color = get_random_color()

            type = self.df_objs["type"].take([i]).item()

            if type == "pathText":
                continue

            points = self.df_objs["rotated_bbox"].take([i]).item()

            x1 = points[0]
            y1 = points[1]
            x2 = points[2]
            y2 = points[3]
            x3 = points[4]
            y3 = points[5]
            x4 = points[6]
            y4 = points[7]

            size = 60
            draw.ellipse([x1 - size / 2, y1 - size / 2, x1 + size // 2, y1 + size // 2], fill=rand_color)
            draw.ellipse([x2 - size / 2, y2 - size / 2, x2 + size // 2, y2 + size // 2], fill=rand_color)
            draw.ellipse([x3 - size / 2, y3 - size / 2, x3 + size // 2, y3 + size // 2], fill=rand_color)
            draw.ellipse([x4 - size / 2, y4 - size / 2, x4 + size // 2, y4 + size // 2], fill=rand_color)

            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=rand_color, width=5)

    # ------------------------------- Bbox scaling -----------------------------------------------------------------------
    def scale_bboxes(self) -> None:
        """
        Function to scale bboxes
        """

        self.df_objs["top"] = self.df_objs["top"]
        self.df_objs["left"] = self.df_objs["left"]
        self.df_objs["width"] = self.df_objs["width"] * self.df_objs["scaleX"]
        self.df_objs["height"] = self.df_objs["height"] * self.df_objs["scaleY"]

    # ------------------------------- Get functions -----------------------------------------------------------------------
    def get_bbox_list(self, exclude_text=False, exclude_basic=False):
        """
        Get list of bounding boxes from template. There is the option to exclude text & basic shape elements.
        """
        all_bboxes = []
        for no_box in range(len(self.df_objs)):
            object_type = self.df_objs["type"].take([no_box]).item()

            if "clipPathMaskId" in self.df_objs.columns:
                clip_path_mask_id = self.df_objs["clipPathMaskId"].take([no_box]).item()
            else:
                clip_path_mask_id = None

            if exclude_text:
                if object_type == "pathText" or type(clip_path_mask_id) == str:
                    continue

            if exclude_basic:
                if object_type == "basicShape":
                    continue

            x1 = int(self.df_objs["left"].take([no_box]).item())
            y1 = int(self.df_objs["top"].take([no_box]).item())
            width = int(self.df_objs["width"].take([no_box]).item())
            height = int(self.df_objs["height"].take([no_box]).item())

            # check for full coverage
            if (
                    width >= self.canvas_width
                    and height >= self.canvas_height
                    and x1 <= 0
                    and y1 <= 0
            ):
                continue

            all_bboxes.append([x1, y1, x1 + width, y1 + height])

        return all_bboxes

    # ------------------------------- Processing -----------------------------------------------------------------------
    def apply_mask_to_image(self, idx, object, image_to_be_masked, masked_image_id):
        """
        Apply a mask to an image
        """
        mask_width = object["width"] * object["scaleX"]
        mask_height = object["height"] * object["scaleY"]
        mask_top = object["top"]
        mask_left = object["left"]

        mask = self.images[idx].resize((round(mask_width), round(mask_height)))

        image_to_be_masked_width = self.df_objs.take([masked_image_id])["width"].item()
        image_to_be_masked_height = self.df_objs.take([masked_image_id])[
            "height"
        ].item()
        image_to_be_masked_top = self.df_objs.take([masked_image_id])["top"].item()
        image_to_be_masked_left = self.df_objs.take([masked_image_id])["left"].item()
        image_to_be_masked_scalex = self.df_objs.take([masked_image_id])[
            "scaleX"
        ].item()
        image_to_be_masked_scaley = self.df_objs.take([masked_image_id])[
            "scaleY"
        ].item()

        image_to_be_masked = image_to_be_masked.resize(
            (
                int(image_to_be_masked_width * image_to_be_masked_scalex),
                int(image_to_be_masked_height * image_to_be_masked_scaley),
            ),
        ).convert("RGBA")

        temp_canvas_image = Image.new(
            "RGBA",
            (self.canvas_width, self.canvas_height),
            (255, 255, 255, 255),
        )
        temp_canvas_mask_image = Image.new(
            "RGBA",
            (self.canvas_width, self.canvas_height),
            (0, 0, 0, 0),
        )

        temp_canvas_image.paste(
            image_to_be_masked,
            (
                round(image_to_be_masked_left),
                round(image_to_be_masked_top),
            ),
            image_to_be_masked,
        )

        temp_canvas_image_np = np.array(temp_canvas_image)
        temp_canvas_image_np = temp_canvas_image_np[:, :, 0:3]

        temp_canvas_mask_image.paste(
            mask,
            (
                round(mask_left),
                round(mask_top),
            ),
            mask,
        )

        temp_canvas_mask_np = np.array(temp_canvas_mask_image)
        temp_canvas_mask_np = temp_canvas_mask_np[:, :, 0:3]

        _, temp_canvas_mask_np = cv2.threshold(
            temp_canvas_mask_np,
            0,
            255,
            cv2.THRESH_BINARY,
        )
        temp_canvas_mask_np = temp_canvas_mask_np[:, :, 0]

        masked_image = cv2.bitwise_and(
            temp_canvas_image_np.astype(np.uint8),
            temp_canvas_image_np.astype(np.uint8),
            mask=temp_canvas_mask_np,
        )
        tmp = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(masked_image)
        rgba = [b, g, r, alpha]
        masked_tr = cv2.merge(rgba, 4)

        masked_image = Image.fromarray(masked_tr.astype("uint8"), "RGBA")

        return masked_image

    def render_template(
            self,
            update_rendered_image=False,
            do_print_angle_text=False,
            debug=False,
    ):
        """
        Function to render a template considering scaling, flipping, rotation and masking of visual elements.

        When elements are rotated the bounding box needs to be recomputed!
        """

        def get_bbox_dimensions(points):
            """
            Function to get bbox dimensions after rotation
            """
            x = points.T[0]
            y = points.T[1]
            max_x = np.max(x)
            min_x = np.min(x)
            min_y = np.min(y)
            max_y = np.max(y)
            return {"width": max_x - min_x, "height": max_y - min_y}

        def rotate_point(point, angle):
            """
            Function to rotate a 2D point
            """
            t = math.radians(angle)
            rotation = np.array(
                [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]],
            )
            return np.dot(rotation, point.T)

        def flip(image, axis="x"):
            """
            Function to flip image
            """
            arr = np.asarray(image)
            axis = 0 if axis == "y" else 1
            arr = np.flip(arr, axis)
            return Image.fromarray(arr.astype(np.uint8))

        self.rendered_image = Image.new(
            "RGBA",
            (self.canvas_width, self.canvas_height),
            (255, 255, 255, 255),
        )

        self.df_objs["rotated_bbox"] = [None] * len(self.df_objs)
        self.df_objs["rotated_bbox"] = self.df_objs["rotated_bbox"].apply(lambda x: [] if pd.isna(x) else x)

        mask_dict = {}
        for idx, object in self.df_objs.iterrows():
            if "clipPathMaskId" in object.keys():
                if type(object["clipPathMaskId"]) == str:
                    mask_dict[object["clipPathMaskId"]] = idx

        for idx, object in self.df_objs.iterrows():
            render_element = True
            angle = object["angle"]

            if "clipPathMaskId" in object.keys():
                if type(object["clipPathMaskId"]) == str:
                    continue

            if object["type"] == "mask":
                masked_image_ind = mask_dict[object["id"]]
                image_to_be_masked = self.images[masked_image_ind]

                masked_image = self.apply_mask_to_image(
                    idx,
                    object,
                    image_to_be_masked,
                    masked_image_ind,
                )

                self.rendered_image.paste(
                    masked_image,
                    (
                        round(0),
                        round(0),
                    ),
                    masked_image,
                )
                render_element = False
                self.df_objs.drop(masked_image_ind, inplace=True)

            if object["type"] == "pathText" and do_print_angle_text:
                print(f"Angle of text element {angle}")

            image = self.images[idx]

            width = object["width"] * object["scaleX"]
            height = object["height"] * object["scaleY"]

            # ----------------------- Compute points --- ---------------------------------------------------------------
            points = np.array(
                [
                    np.array([-width / 2, -height / 2]),  # top-left
                    np.array([width / 2, -height / 2]),  # top-right
                    np.array([width / 2, height / 2]),  # bottom-right
                    np.array([-width / 2, height / 2]),  # bottom-left
                ],
            )

            if debug:
                print("Points before rotation: ", points)

            # ----------------------- Rotate points -------------------------------------------------------------------
            points = [rotate_point(point, angle) for point in points]

            if debug:
                print("Points after rotation: ", points)

            # ----------------------- Compute offset vectors ------------------------------------------------------------
            bbox_dimensions = get_bbox_dimensions(np.array(points))

            bbox_top_left = np.array([-bbox_dimensions["width"] / 2, -bbox_dimensions["height"] / 2], )
            offset_top_left = bbox_top_left - points[0]

            top_left = np.array([object['left'], object['top']])
            points += top_left - points[0]

            new_width = bbox_dimensions["width"]
            new_height = bbox_dimensions["height"]

            # ----------------------- Flip images ----------------------------------------------------------------------
            if object["type"] != "pathText":
                if object["flipX"]:
                    image = flip(image, "x")
                if object["flipY"]:
                    image = flip(image, "y")

            # ----------------------- Update df_objs dataframe ---------------------------------------------------------
            if (
                    not math.isnan(object["left"])
                    and not math.isnan(object["top"])
                    and not math.isnan(offset_top_left[0])
                    and not math.isnan(offset_top_left[1])
            ):
                self.df_objs.at[idx, "left"] = round(object["left"] + offset_top_left[0])
                self.df_objs.at[idx, "top"] = round(object["top"] + offset_top_left[1])
                self.df_objs.at[idx, "width"] = new_width
                self.df_objs.at[idx, "height"] = new_height

                self.df_objs.at[idx, "rotated_bbox"] = points.flatten().tolist()

            else:
                self.is_invalid = True

            if not self.is_invalid and render_element:
                # ----------------------- Scale & Rotate images ------------------------------------------------------------
                if (
                        object["type"] != "pathText"
                        and self.images[idx] is not None
                        and round(width) > 0
                        and round(height) > 0
                ):
                    image = image.convert("RGBA")
                    image = image.resize((round(width), round(height)))
                    image = image.rotate(-object["angle"], expand=True)
                    self.images[idx] = image

                    # ----------------------- Update rendered image ------------------------------------------------------------
                    if update_rendered_image:
                        self.rendered_image.paste(
                            image,
                            (
                                round(object["left"] + offset_top_left[0]),
                                round(object["top"] + offset_top_left[1]),
                            ),
                            image,
                        )

    def simple_render(self):

        def flip(image, axis="x"):
            """
            Function to flip image
            """
            arr = np.asarray(image)
            axis = 0 if axis == "y" else 1
            arr = np.flip(arr, axis)
            return Image.fromarray(arr.astype(np.uint8))

        self.rendered_image = Image.new(
            "RGBA",
            (self.canvas_width, self.canvas_height),
            (255, 255, 255, 255),
        )

        mask_dict = {}
        for idx, object in self.df_objs.iterrows():
            if "clipPathMaskId" in object.keys():
                if type(object["clipPathMaskId"]) == str:
                    mask_dict[object["clipPathMaskId"]] = idx

        for idx, object in self.df_objs.iterrows():
            render_element = True

            image = self.images[idx]

            width = object["width"] * object["scaleX"]
            height = object["height"] * object["scaleY"]

            # ----------------------- Flip images ----------------------------------------------------------------------
            if object["type"] != "pathText":
                if object["flipX"]:
                    image = flip(image, "x")
                if object["flipY"]:
                    image = flip(image, "y")

            # ----------------------- Update df_objs dataframe ---------------------------------------------------------
            if (
                    not math.isnan(object["left"])
                    and not math.isnan(object["top"])
            ):
                self.df_objs.at[idx, "left"] = round(object["left"])
                self.df_objs.at[idx, "top"] = round(object["top"])
                self.df_objs.at[idx, "width"] = width
                self.df_objs.at[idx, "height"] = height

            else:
                self.is_invalid = True

            if not self.is_invalid and render_element:
                # ----------------------- Scale & Rotate images ------------------------------------------------------------
                if (
                        object["type"] != "pathText"
                        and self.images[idx] is not None
                        and round(width) > 0
                        and round(height) > 0
                ):
                    image = image.convert("RGBA")
                    image = image.resize((round(width), round(height)))
                    image = image.rotate(-object["angle"], expand=True)
                    self.images[idx] = image

                    # ----------------------- Update rendered image ------------------------------------------------------------
                    self.rendered_image.paste(
                        image,
                        (
                            round(object["left"]),
                            round(object["top"]),
                        ),
                        image,
                    )

    def render_and_visualize(self, dst_image="preview", visualize_bbox=True, visualize_rotated_bbox=False) -> bool:
        """
        Function to render a template and visualize the result, returns True if successful, false otherwise.

        """
        self.load_image_data()
        if dst_image == "preview":
            self.load_preview_image()

        self.render_template(update_rendered_image=True)

        if self.is_invalid:
            return False

        if dst_image == "preview":
            if visualize_bbox:
                self.draw_on_preview_image()
                return True
        else:
            if visualize_bbox:
                self.draw_on_rendered_image()
            if visualize_rotated_bbox:
                self.visualize_rotated_bbox()
            return True
