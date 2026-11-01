"""Image/pose batches for PIS, adapted from CompVis VUNet.

Only load dataset pickle files from a trusted source. Match maps are explicit
JSON objects mapping source identity strings to distinct donor identities.
"""

from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
import json
import math
import os
import pickle

import cv2
import numpy as np
import PIL.Image


n_boxes = 8
DEFAULT_JOINT_ORDER = [
    "rankle", "rknee", "rhip", "lhip", "lknee", "lankle", "rwrist", "relbow",
    "rshoulder", "lshoulder", "lelbow", "lwrist", "neck", "cnose", "leye",
    "reye", "lear", "rear",
]
RANDOM_KEYS = ["imgs", "joints", "norm_imgs", "norm_joints", "pids",
               "imgs_name", "joints_coordinates"]
MATCH_KEYS = ["imgs", "joints", "norm_imgs", "norm_joints", "imgs_name",
              "index_list"]


class BufferedWrapper:
    """Prefetch one batch; release the worker when a finite flow is exhausted."""

    def __init__(self, gen):
        self.gen = gen
        self.n = gen.n
        self.pids = gen.pids
        self.n_groups = gen.n_groups
        self.pool = ThreadPool(1)
        self.buffer_ = self.pool.apply_async(next, (self.gen,))
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            result = self.buffer_.get()
        except BaseException:
            self.close()
            raise
        if self.gen.sample == "match" and self.gen.p_id >= len(self.pids):
            self.close()
        else:
            self.buffer_ = self.pool.apply_async(next, (self.gen,))
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _basename(path):
    return os.fspath(path).replace("\\", "/").rsplit("/", 1)[-1]


def _identity(path):
    name = _basename(path)
    identity, separator, _ = name.partition("_")
    if not separator or not identity:
        raise ValueError("Image filenames must begin with an identity and '_': " + name)
    return identity


def load_img(path, target_size):
    """Load an RGB/grayscale uint8 image; target_size is (height, width, channels)."""
    with PIL.Image.open(path) as image:
        image = image.convert("L" if target_size[2] == 1 else "RGB")
        size = (target_size[1], target_size[0])
        if image.size != size:
            image = image.resize(size, resample=PIL.Image.Resampling.BILINEAR)
        result = np.asarray(image, dtype=np.uint8)
    return result[..., None] if result.ndim == 2 else result


def preprocess(x):
    """Convert uint8 pixels to float32 in [-1, 1]."""
    return np.asarray(x, dtype=np.float32) / 127.5 - 1.0


def postprocess(x):
    """Convert [-1, 1] pixels to clipped uint8."""
    return np.clip((np.asarray(x) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def tile(images, rows, cols):
    canvas = np.zeros((rows * images.shape[1], cols * images.shape[2],
                       images.shape[3]), dtype=images.dtype)
    for index, image in enumerate(images[:rows * cols]):
        row, col = divmod(index, cols)
        canvas[row * image.shape[0]:(row + 1) * image.shape[0],
               col * image.shape[1]:(col + 1) * image.shape[1]] = image
    return canvas


def plot_batch(images, out_path, key=None):
    if images.shape[3] > 3:
        images = images[:, :, :, np.random.choice(images.shape[3], size=3)]
    images = postprocess(images)
    cols = 8 if key == "interpolate_2x" else math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    canvas = np.squeeze(tile(images, rows, cols))
    PIL.Image.fromarray(canvas).save(out_path)
    return canvas


def save_dataset(X, out_path, image_paths, nx=1, id_num=9, identity_offset=1500):
    """Save center-half crops directly under out_path using shifted source IDs.

    nx and id_num remain accepted for compatibility; each output requires its
    own source filename, including when callers generate multiple variants.
    """
    images = np.asarray(X)
    names = list(image_paths)
    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[2] < 2:
        raise ValueError("Generated images must have shape [N, H, W, 3], W >= 2")
    if len(images) != len(names):
        raise ValueError("Generated image count must match the number of source filenames")
    if not np.isfinite(images).all():
        raise ValueError("Generated images must contain only finite values")
    if not isinstance(identity_offset, (int, np.integer)):
        raise ValueError("identity_offset must be an integer")
    output_names = []
    for path in names:
        name = _basename(path)
        identity = _identity(name)
        try:
            shifted = int(identity) + identity_offset
        except ValueError as error:
            raise ValueError("Output identity must be numeric: " + identity) from error
        output_identity = identity if identity_offset == 0 else str(shifted)
        output_names.append(output_identity + name[len(identity):])
    if len(set(output_names)) != len(output_names):
        raise ValueError("Source filenames produce duplicate output names")
    output = Path(out_path)
    for name in output_names:
        target = output / name
        if target.exists() or target.is_symlink():
            raise FileExistsError("Refusing to overwrite existing output: " + str(target))
    output.mkdir(parents=True, exist_ok=True)
    pixels = postprocess(images)
    width = images.shape[2]
    crop_width = width // 2
    start = (width - crop_width) // 2
    for image, name in zip(pixels, output_names):
        PIL.Image.fromarray(image[:, start:start + crop_width, :]).save(output / name)


def make_joint_img(img_shape, jo, joints):
    # three channels: left, right, center
    scale_factor = img_shape[1] / 128
    thickness = max(1, int(3 * scale_factor))
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part),:] for part in body]])
    if np.min(body_pts) >= 0:
        body_pts = np.int_(body_pts)
        cv2.fillPoly(imgs[2], body_pts, 255)

    right_lines = [
            ("rankle", "rknee"),
            ("rknee", "rhip"),
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist")]
    for line in right_lines:
        l = [jo.index(line[0]), jo.index(line[1])]
        if np.min(joints[l]) >= 0:
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

    left_lines = [
            ("lankle", "lknee"),
            ("lknee", "lhip"),
        ("lhip", "lshoulder"),
        ("lshoulder", "lelbow"),
        ("lelbow", "lwrist")]
    for line in left_lines:
        l = [jo.index(line[0]), jo.index(line[1])]
        if np.min(joints[l]) >= 0:
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[1], a, b, color=255, thickness=thickness)

    rs = joints[jo.index("rshoulder")]
    ls = joints[jo.index("lshoulder")]
    cn = joints[jo.index("cnose")]
    neck = 0.5 * (rs + ls)
    a = tuple(np.int_(neck))
    b = tuple(np.int_(cn))
    if np.min(a) >= 0 and np.min(b) >= 0:
        cv2.line(imgs[0], a, b, color=127, thickness=thickness)
        cv2.line(imgs[1], a, b, color=127, thickness=thickness)

    cn = tuple(np.int_(cn))
    leye = tuple(np.int_(joints[jo.index("leye")]))
    reye = tuple(np.int_(joints[jo.index("reye")]))
    if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
        cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)
        cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)

    img = np.stack(imgs, axis=-1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis=-1)[:, :, None]
    return img


def valid_joints(*joints):
    j = np.stack(joints)
    return (j >= 0).all()


def get_crop(bpart, joints, jo, wh, o_w, o_h, ar=1.0):
    bpart_indices = [jo.index(b) for b in bpart]
    part_src = np.float32(joints[bpart_indices])

    # fall backs
    if not valid_joints(part_src):
        if bpart[0] == "lhip" and bpart[1] == "lknee":
            bpart = ["lhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "rhip" and bpart[1] == "rknee":
            bpart = ["rhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
            bpart = ["lshoulder", "rshoulder", "rshoulder"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])

    if not valid_joints(part_src):
            return None

    if part_src.shape[0] == 1:
        # leg fallback
        a = part_src[0]
        b = np.float32([a[0], o_h - 1])
        part_src = np.float32([a, b])

    if part_src.shape[0] == 4:
        pass
    elif part_src.shape[0] == 3:
        # lshoulder, rshoulder, cnose
        if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal
            part_src = np.float32([a, b, c, d])
        else:
            assert bpart == ["lshoulder", "rshoulder", "cnose"]
            neck = 0.5*(part_src[0] + part_src[1])
            neck_to_nose = part_src[2] - neck
            part_src = np.float32([neck + 2*neck_to_nose, neck])

            # segment box
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = 1.0 / 2.0
            a = part_src[0] + alpha * normal
            b = part_src[0] - alpha * normal
            c = part_src[1] - alpha * normal
            d = part_src[1] + alpha * normal
            # part_src = np.float32([a,b,c,d])
            part_src = np.float32([b, c, d, a])
    else:
        assert part_src.shape[0] == 2

        segment = part_src[1] - part_src[0]
        normal = np.array([-segment[1], segment[0]])
        alpha = ar / 2.0
        a = part_src[0] + alpha * normal
        b = part_src[0] - alpha * normal
        c = part_src[1] - alpha * normal
        d = part_src[1] + alpha * normal
        part_src = np.float32([a, b, c, d])

    dst = np.float32([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    part_dst = np.float32(wh * dst)

    M = cv2.getPerspectiveTransform(part_src, part_dst)
    return M


def normalize(imgs, coords, stickmen, jo, box_factor):
    out_imgs = list()
    out_stickmen = list()

    bs = len(imgs)
    for i in range(bs):
        img = imgs[i]
        joints = coords[i]
        stickman = stickmen[i]

        h, w = img.shape[:2]
        o_h = h
        o_w = w
        h = h // 2 ** box_factor
        w = w // 2 ** box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
            ["lshoulder", "lhip", "rhip", "rshoulder"],
            ["lshoulder", "rshoulder", "cnose"],
            ["lshoulder", "lelbow"],
            ["lelbow", "lwrist"],
            ["rshoulder", "relbow"],
            ["relbow", "rwrist"],
            ["lhip", "lknee"],
            ["rhip", "rknee"]]
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()
        for bpart in bparts:
            part_img = np.zeros((h, w, 3), dtype=np.float32)
            part_stickman = np.zeros((h, w, 3), dtype=np.float32)
            M = get_crop(bpart, joints, jo, wh, o_w, o_h, ar)

            if M is not None:
                part_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                part_stickman = cv2.warpPerspective(stickman, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)
        img = np.concatenate(part_imgs, axis=2)
        stickman = np.concatenate(part_stickmen, axis=2)

        out_imgs.append(img)
        out_stickmen.append(stickman)
    out_imgs = np.stack(out_imgs)
    out_stickmen = np.stack(out_stickmen)
    return out_imgs, out_stickmen



class IndexFlow:
    """Infinite random training batches or one finite pass over matched identities."""

    def __init__(self, shape, index_path, train, box_factor, sample="random", nx=1,
                 fill_batches=True, shuffle=True, return_keys=None,
                 match_path=None, batch_id_size=27):
        self.shape = list(shape)
        if (len(self.shape) != 4 or any(not isinstance(value, (int, np.integer))
                                      or value <= 0 for value in self.shape)
                or self.shape[-1] != 3):
            raise ValueError("shape must be positive integer [batch, height, width, 3]")
        if sample not in ("random", "match"):
            raise ValueError("sample must be 'random' or 'match'")
        if not isinstance(box_factor, int) or box_factor < 0:
            raise ValueError("box_factor must be a nonnegative integer")
        if min(self.shape[1:3]) // 2 ** box_factor < 1:
            raise ValueError("box_factor produces empty body crops")
        if not isinstance(batch_id_size, int) or batch_id_size < 1:
            raise ValueError("batch_id_size must be a positive integer")
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.basepath = Path(index_path).parent
        self.train = bool(train)
        self.fill_batches = fill_batches
        self.sample = sample
        self.nx = nx
        self.batch_id_size = batch_id_size
        self.shuffle_ = bool(shuffle) and sample == "random"
        with open(index_path, "rb") as handle:
            self.index = pickle.load(handle)
        if not isinstance(self.index, dict):
            raise ValueError("Dataset index must be a dictionary")
        missing = {"imgs", "joints", "train"} - self.index.keys()
        if missing:
            raise ValueError("Dataset index is missing: " + ", ".join(sorted(missing)))
        self.jo = list(self.index.get("joint_order", DEFAULT_JOINT_ORDER))
        if (len(self.jo) != len(set(self.jo))
                or not set(DEFAULT_JOINT_ORDER).issubset(self.jo)):
            raise ValueError("joint_order must contain all 18 required joint names without duplicates")
        count = len(self.index["imgs"])
        joints = np.asarray(self.index["joints"], dtype=np.float32)
        train_flags = np.asarray(self.index["train"])
        if joints.shape != (count, len(self.jo), 2):
            raise ValueError("joints must have shape [number of images, joint_order length, 2]")
        if train_flags.shape != (count,):
            raise ValueError("train must contain one split flag per image")
        if not np.isfinite(joints).all() or np.any(joints > 1):
            raise ValueError("joints must be finite normalized xy coordinates; negative values mark occlusion")
        self.index["joints"] = joints * np.array(self.img_shape[1::-1], dtype=np.float32)
        self.index["train"] = train_flags
        self.indices = np.array([i for i in range(count) if self._filter(i)], dtype=np.int64)
        self.n = len(self.indices)
        if not self.n:
            split = "training" if self.train else "validation"
            raise ValueError("No valid images in the " + split + " split; check split flags and shoulder/hip joints")
        self.index_dic = defaultdict(list)
        for index in self.indices:
            self.index_dic[_identity(self.index["imgs"][index])].append(int(index))
        self.pids = list(self.index_dic)
        self.pid2label = {pid: label for label, pid in enumerate(self.pids)}
        self.n_groups = math.ceil(len(self.pids) / self.batch_id_size)
        self.p_id = 0
        self.return_keys = list(return_keys) if return_keys is not None else (
            RANDOM_KEYS.copy() if sample == "random" else MATCH_KEYS.copy())
        if sample == "match":
            if nx != 1:
                raise ValueError("match sampling uses exactly one donor per identity (nx=1)")
            if match_path is None:
                raise ValueError("match sampling requires an explicit JSON match_path")
            with open(match_path, encoding="utf-8") as handle:
                self.id2oid = json.load(handle)
            if not isinstance(self.id2oid, dict):
                raise ValueError("match map must be a JSON object mapping source IDs to donor IDs")
            for identity in self.pids:
                donor = self.id2oid.get(identity)
                if not isinstance(donor, str) or donor not in self.index_dic:
                    raise ValueError("Missing or unavailable donor in match map for identity " + identity)
                if donor == identity:
                    raise ValueError("match donor must differ from source identity " + identity)
        self.shuffle()

    def __iter__(self):
        return self

    def _filter(self, index):
        if self.index["train"][index] != self.train:
            return False
        required = [self.jo.index(name) for name in
                    ("lshoulder", "rshoulder", "lhip", "rhip")]
        return valid_joints(self.index["joints"][index, required])

    def match_sampler(self):
        if self.p_id >= len(self.pids):
            raise StopIteration
        identities = self.pids[self.p_id:self.p_id + self.batch_id_size]
        sources, donors, lengths = [], [], []
        for identity in identities:
            indices = self.index_dic[identity]
            sources.extend(indices)
            lengths.append(len(indices))
            donors.append(self.index_dic[self.id2oid[identity]][0])
        self.p_id += len(identities)
        return sources, donors, lengths

    def _random_indices(self):
        remaining = self.batch_size
        parts = []
        while remaining:
            end = min(self.batch_start + remaining, self.n)
            part = self.indices[self.batch_start:end].copy()
            parts.append(part)
            remaining -= len(part)
            self.batch_start = end
            if self.batch_start == self.n:
                self.shuffle()
                if not self.fill_batches:
                    break
        return np.concatenate(parts)

    def __next__(self):
        batch = {}
        if self.sample == "match":
            sources, donors, batch["index_list"] = self.match_sampler()
            indices = np.asarray(donors + sources, dtype=np.int64)
        else:
            indices = self._random_indices()
        batch["indices"] = indices
        names = [self.index["imgs"][index] for index in indices]
        batch["imgs_name"] = np.asarray(names)
        batch["pids"] = np.asarray([self.pid2label[_identity(name)] for name in names], dtype=np.int64)
        batch["imgs"] = preprocess(np.stack([
            load_img(self.basepath / os.fspath(name).replace("\\", "/"), self.img_shape)
            for name in names
        ]))
        coordinates = self.index["joints"][indices]
        batch["joints_coordinates"] = coordinates
        batch["joints"] = preprocess(np.stack([
            make_joint_img(self.img_shape, self.jo, joints) for joints in coordinates
        ]))
        batch["norm_imgs"], batch["norm_joints"] = normalize(
            batch["imgs"], coordinates, batch["joints"], self.jo, self.box_factor)
        return [batch[key] for key in self.return_keys]

    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


def get_batches(shape, index_path, train, box_factor, sample="random", nx=1,
                fill_batches=True, shuffle=True, return_keys=None,
                match_path=None, batch_id_size=27):
    """Return a closeable prefetch wrapper; use IndexFlow directly for generation."""
    return BufferedWrapper(IndexFlow(
        shape, index_path, train, box_factor, sample, nx, fill_batches, shuffle,
        return_keys, match_path, batch_id_size))
