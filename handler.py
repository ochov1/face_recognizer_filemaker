import base64
import binascii
import cv2
import numpy as np
import os
import runpod
import torch
import requests
from urllib.parse import urlparse

from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1

class RobustFaceEmbedding:
    def __init__(self):
        self.insightface_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.insightface_model.prepare(ctx_id=0, det_size=(640, 640))
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

    def get_combined_embedding(self, image):
        """Get combined embedding from both models with optional caching."""
        if isinstance(image, list):
            return self._batch_process_embeddings(image)
        embedding = self._compute_embedding(image)
        return embedding

    def _batch_process_embeddings(self, images, batch_size=32):
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = self._compute_embedding_batch(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    def _compute_embedding_batch(self, images):
        embeddings = []
        for image in images:
            embedding = self._compute_embedding(image)
            embeddings.append(embedding)
        return embeddings

    def _compute_embedding(self, image):
        """
        Compute face embedding using both InsightFace and FaceNet models.

        Args:
            image: numpy array in BGR format (H, W, 3)

        Returns:
            Combined normalized embedding vector
        """
        try:
            # Ensure image is numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError("Input image must be a numpy array")

            # Upscale tiny inputs so the detector sees more detail
            h, w = image.shape[:2]
            
            if max(h, w) < 640:
                image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)

            # Prepare image for InsightFace
            faces = self.insightface_model.get(image)
            if not faces:
                raise ValueError("No face detected in the image")
                # Save the image that failed detection for debugging purposes.
                debug_dir = os.path.join(os.path.expanduser("~"), "failed_detections")
                os.makedirs(debug_dir, exist_ok=True)
                import time
                timestamp = int(time.time() * 1000)
                failed_image_path = os.path.join(debug_dir, f"no_face_detected_{timestamp}.jpg")
                cv2.imwrite(failed_image_path, image) # Save the image before raising the error
                raise ValueError(f"No face detected in the image. The problematic image has been saved to: {failed_image_path}")

            insightface_embedding = faces[0].embedding
            # Prepare image for FaceNet
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize using OpenCV
            resized_image = cv2.resize(rgb_image, (160, 160))

            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(resized_image).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
            img_tensor = (img_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            # Get FaceNet embedding
            with torch.no_grad():
                facenet_embedding = self.facenet_model(img_tensor).cpu().numpy().flatten()
            # Normalize individual embeddings
            insightface_embedding = insightface_embedding / np.linalg.norm(insightface_embedding)
            facenet_embedding = facenet_embedding / np.linalg.norm(facenet_embedding)
            # Combine embeddings with weights
            combined_embedding = np.concatenate([
                insightface_embedding * 0.6,  # 60% weight to InsightFace
                facenet_embedding * 0.4       # 40% weight to FaceNet
            ])
            # Normalize final embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            return combined_embedding

        except Exception as e:
            raise ValueError(f"Error computing embedding: {str(e)}")

def read_image_from_url(image_url: str):
    response = requests.get(image_url)
    arr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def read_image_from_base64(image_b64: str):
    """Decode a base64-encoded image string into an OpenCV image."""
    if image_b64.startswith("data:"):
        # Strip data URL header if present
        try:
            image_b64 = image_b64.split(",", 1)[1]
            image_b64 = image_b64.replace("\n", "").replace("\r", "").strip()
        except IndexError as exc:
            raise ValueError("Invalid data URL format") from exc

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 image data") from exc

    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode base64 image data")
    return image


def read_image(image_input):
    """Support http(s) URLs or raw/data-URL base64 strings, including lists."""
    if isinstance(image_input, list):
        return [read_image(img) for img in image_input]

    if not isinstance(image_input, str):
        raise ValueError("Image input must be a string or list of strings")

    # Explicitly handle data URLs
    if image_input.startswith("data:"):
        return read_image_from_base64(image_input)

    parsed = urlparse(image_input)
    # Only treat as URL when we have both scheme and netloc
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return read_image_from_url(image_input)

    # Fallback: try to decode as base64 (raw string or malformed URL)
    return read_image_from_base64(image_input)

def parse_insightface_output(output):
    """
    Parse the output from InsightFace model to extract face embeddings.

    Args:
        output: List of detected faces with embeddings.

    Returns:
        List of detected faces with metrics as lists.
    """
    if not output:
        return []
    faces = []
    for face in output:
        face_dict = {
            'bbox': face.bbox.tolist(),
            'kps': face.kps.tolist(),
            'det_score': float(face.det_score),
            'landmark_3d_68': face.landmark_3d_68.tolist() if hasattr(face, 'landmark_3d_68') else None,
            'pose': face.pose.tolist() if hasattr(face, 'pose') else None,
            'landmark_2d_106': face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None,
            'gender': float(face.gender),
            'age': float(face.age),
            'embedding': face.embedding.tolist(),
        }
        faces.append(face_dict)
    return faces

def handler(event):
    event = event.get("input", {})
    image_input = event.get("image", "")
    image = read_image(image_input)
    purpose = event.get("purpose", "insightface")
    face_embedding = RobustFaceEmbedding()
    if purpose=="combined_embedding":
        result = face_embedding.get_combined_embedding(image)
        if isinstance(result, list):
            result = [embedding.tolist() for embedding in result]
        else: result = result.tolist()
    elif purpose=="insightface":
        if isinstance(image, list):
            result = [face_embedding.insightface_model.get(img) for img in image]
            result = [parse_insightface_output(img) for img in result]
        else:
            result = face_embedding.insightface_model.get(image)
            result = parse_insightface_output(result)
    return result

if __name__ == "__main__":
    pass
