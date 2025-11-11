"""
ComfyUI Qwen Frame Selector Node
Author: eddy
Description: Intelligent video frame selection using Qwen VL vision model
"""

import os
import uuid
import json
import requests
import base64
import subprocess
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    class folder_paths:
        @staticmethod
        def get_temp_directory():
            import tempfile
            return tempfile.gettempdir()


class QwenFrameSelector:
    """
    Intelligent video frame selector using Qwen VL model
    Analyzes video frames and selects the best quality frames based on multiple criteria
    """
    
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "qwen/qwen3-vl-235b-a22b-thinking"
        self.temp_dir = folder_paths.get_temp_directory() if FOLDER_PATHS_AVAILABLE else "/tmp"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "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",
                    "multiline": False,
                    "placeholder": "OpenRouter API Key"
                }),
                "sample_rate": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Extract 1 frame every N frames"
                }),
                "selection_criteria": ([
                    "quality_focused",      # Focus on sharpness and technical quality
                    "aesthetic_focused",    # Focus on composition and beauty
                    "balanced",            # Balance all criteria
                    "content_diversity",   # Maximize content variation
                    "keyframe_detection"   # Detect significant scene changes
                ], {
                    "default": "balanced"
                }),
                "top_percent": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 100.0,
                    "step": 5.0,
                    "display": "number",
                    "tooltip": "Keep top X% of frames"
                }),
                "min_score": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "Minimum quality score (0-10)"
                }),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Input video for frame extraction and analysis"}),
                "images": ("IMAGE", {"tooltip": "Input images for direct analysis (alternative to video)"}),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Maximum frames to select (0 = unlimited)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("selected_frames", "scores_json", "summary")
    FUNCTION = "select_frames"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True
    
    def extract_all_frames(self, video_path: str, sample_rate: int = 30) -> List[Path]:
        """Extract frames from video at specified sample rate"""
        if not os.path.isabs(video_path) and FOLDER_PATHS_AVAILABLE:
            try:
                input_dir = folder_paths.get_input_directory()
                full_path = os.path.join(input_dir, video_path)
                if os.path.exists(full_path):
                    video_path = full_path
            except Exception:
                pass
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create temp directory for frames
        frames_dir = Path(self.temp_dir) / "qwen_frame_selector" / uuid.uuid4().hex
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[QwenFrameSelector] Extracting frames to: {frames_dir}")
        
        # Extract frames using ffmpeg with sample rate
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{sample_rate}))',
            '-vsync', '0',
            '-q:v', '2',
            '-f', 'image2',
            str(frames_dir / 'frame_%06d.jpg')
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg frame extraction failed: {e.stderr.decode()}")
        
        # Get list of extracted frames
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        print(f"[QwenFrameSelector] Extracted {len(frame_paths)} frames")
        
        return frame_paths
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    
    def get_evaluation_prompt(self, criteria: str) -> str:
        """Get frame evaluation prompt based on criteria"""
        base_prompt = """Evaluate this video frame on the following dimensions (score 0-10 for each):

1. SHARPNESS: Image clarity, focus quality, motion blur
2. COMPOSITION: Rule of thirds, balance, framing, visual interest
3. AESTHETIC: Color harmony, lighting, artistic value
4. TECHNICAL: Exposure, contrast, color accuracy, noise level
5. CONTENT: Information richness, key moments, expression/action quality

"""
        
        criteria_instructions = {
            "quality_focused": """
Prioritize technical quality metrics (sharpness, exposure, noise).
Give higher weight to sharp, well-exposed, technically sound frames.
""",
            "aesthetic_focused": """
Prioritize aesthetic and compositional qualities.
Look for beautiful lighting, harmonious colors, interesting compositions.
""",
            "balanced": """
Give equal weight to all evaluation dimensions.
Select frames that perform well across all criteria.
""",
            "content_diversity": """
Prioritize frames with unique content or different scenes.
Prefer frames that show variety in subject, action, or setting.
""",
            "keyframe_detection": """
Focus on frames showing significant changes or key moments.
Look for scene transitions, important actions, peak expressions.
"""
        }
        
        output_format = """
Respond ONLY with a JSON object in this exact format (no extra text):
{
    "sharpness": 8.5,
    "composition": 7.0,
    "aesthetic": 9.0,
    "technical": 8.0,
    "content": 7.5,
    "overall": 8.0,
    "reason": "Brief explanation of scores"
}
"""
        
        return base_prompt + criteria_instructions.get(criteria, criteria_instructions["balanced"]) + output_format
    
    def evaluate_frame(self, frame_path: Path, criteria: str) -> Dict:
        """Evaluate a single frame using Qwen VL"""
        prompt = self.get_evaluation_prompt(criteria)
        base64_image = self.encode_image_to_base64(frame_path)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert image quality assessor. Provide objective, numerical evaluations."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON from response
                # Handle markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                scores = json.loads(content)
                return scores
            else:
                print(f"[QwenFrameSelector] API error for frame {frame_path.name}: {response.status_code}")
                # Return default low scores on error
                return {
                    "sharpness": 5.0,
                    "composition": 5.0,
                    "aesthetic": 5.0,
                    "technical": 5.0,
                    "content": 5.0,
                    "overall": 5.0,
                    "reason": f"API error: {response.status_code}"
                }
        except Exception as e:
            print(f"[QwenFrameSelector] Error evaluating frame {frame_path.name}: {e}")
            return {
                "sharpness": 5.0,
                "composition": 5.0,
                "aesthetic": 5.0,
                "technical": 5.0,
                "content": 5.0,
                "overall": 5.0,
                "reason": f"Evaluation error: {str(e)}"
            }
    
    def select_best_frames(
        self,
        frame_scores: List[Dict],
        top_percent: float,
        min_score: float,
        max_frames: int
    ) -> List[int]:
        """Select best frames based on scores and criteria"""
        # Filter by minimum score
        valid_frames = [
            (idx, score) for idx, score in enumerate(frame_scores)
            if score.get('overall', 0) >= min_score
        ]
        
        if not valid_frames:
            print(f"[QwenFrameSelector] Warning: No frames meet minimum score {min_score}")
            # Take top frames even if below threshold
            valid_frames = [(idx, score) for idx, score in enumerate(frame_scores)]
        
        # Sort by overall score
        valid_frames.sort(key=lambda x: x[1].get('overall', 0), reverse=True)
        
        # Calculate how many frames to keep
        target_count = int(len(frame_scores) * (top_percent / 100.0))
        if max_frames > 0:
            target_count = min(target_count, max_frames)
        
        # Select top N frames
        selected_indices = [idx for idx, _ in valid_frames[:target_count]]
        selected_indices.sort()  # Sort by original order
        
        return selected_indices
    
    def load_frames_as_tensor(self, frame_paths: List[Path]) -> torch.Tensor:
        """Load frames as tensor for ComfyUI"""
        images = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
        
        if images:
            return torch.from_numpy(np.array(images))
        else:
            return torch.zeros((1, 64, 64, 3))
    
    def save_images_to_temp(self, images_tensor: torch.Tensor) -> List[Path]:
        """Save IMAGE tensor to temporary files for analysis"""
        frames_dir = Path(self.temp_dir) / "qwen_frame_selector" / uuid.uuid4().hex
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        # images_tensor shape: [B, H, W, C]
        for idx, image in enumerate(images_tensor):
            # Convert from [0,1] float to [0,255] uint8
            img_array = (image.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            output_file = frames_dir / f"image_{idx:06d}.jpg"
            img.save(output_file, quality=95)
            frame_paths.append(output_file)
        
        print(f"[QwenFrameSelector] Saved {len(frame_paths)} images to temp")
        return frame_paths
    
    def select_frames(
        self,
        api_key: str,
        sample_rate: int,
        selection_criteria: str,
        top_percent: float,
        min_score: float,
        video=None,
        images=None,
        max_frames: int = 0,
        unique_id=None
    ):
        """Main function to select best frames from video"""
        try:
            # Update API key
            if api_key and api_key.strip():
                self.api_key = api_key.strip()
            
            # Determine input mode: video or images
            all_frames = []
            
            if images is not None:
                # Image mode: save images to temp and use them directly
                print(f"[QwenFrameSelector] " + "="*60)
                print(f"[QwenFrameSelector] Processing images (Image Mode)")
                print(f"[QwenFrameSelector] Input images count: {images.shape[0]}")
                print(f"[QwenFrameSelector] Selection criteria: {selection_criteria}")
                print(f"[QwenFrameSelector] " + "="*60)
                
                all_frames = self.save_images_to_temp(images)
                
            elif video is not None:
                # Video mode: extract frames from video
                resolved_path = None
                if isinstance(video, str) and video.strip():
                    resolved_path = video.strip()
                elif hasattr(video, "save_to"):
                    temp_dir = os.path.join(self.temp_dir, "qwen_frame_selector_videos")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_name = f"video_{uuid.uuid4().hex}.mp4"
                    temp_path = os.path.join(temp_dir, temp_name)
                    video.save_to(temp_path)
                    resolved_path = temp_path
                else:
                    raise Exception("Unsupported video input type")
                
                print(f"[QwenFrameSelector] " + "="*60)
                print(f"[QwenFrameSelector] Processing video (Video Mode)")
                print(f"[QwenFrameSelector] Video: {resolved_path}")
                print(f"[QwenFrameSelector] Sample rate: 1 frame every {sample_rate} frames")
                print(f"[QwenFrameSelector] Selection criteria: {selection_criteria}")
                print(f"[QwenFrameSelector] " + "="*60)
                
                all_frames = self.extract_all_frames(resolved_path, sample_rate)
            else:
                raise Exception("No input provided! Please connect either a VIDEO or IMAGE input.")
            
            if not all_frames:
                error_msg = "No frames extracted from video"
                return {
                    "ui": {"text": [error_msg]},
                    "result": (torch.zeros((1, 64, 64, 3)), "{}", error_msg)
                }
            
            print(f"[QwenFrameSelector] Evaluating {len(all_frames)} frames...")
            
            # Evaluate each frame
            frame_scores = []
            for idx, frame_path in enumerate(all_frames):
                print(f"[QwenFrameSelector] Evaluating frame {idx+1}/{len(all_frames)}: {frame_path.name}")
                scores = self.evaluate_frame(frame_path, selection_criteria)
                scores['frame_index'] = idx
                scores['frame_path'] = str(frame_path)
                frame_scores.append(scores)
            
            # Select best frames
            selected_indices = self.select_best_frames(
                frame_scores,
                top_percent,
                min_score,
                max_frames
            )
            
            print(f"[QwenFrameSelector] Selected {len(selected_indices)} frames out of {len(all_frames)}")
            
            # Load selected frames as tensor
            selected_frame_paths = [all_frames[idx] for idx in selected_indices]
            frames_tensor = self.load_frames_as_tensor(selected_frame_paths)
            
            # Prepare scores JSON
            selected_scores = [frame_scores[idx] for idx in selected_indices]
            scores_json = json.dumps(selected_scores, indent=2)
            
            # Create summary
            avg_score = sum(s.get('overall', 0) for s in selected_scores) / len(selected_scores)
            summary = f"""Frame Selection Summary:
- Total frames extracted: {len(all_frames)}
- Frames selected: {len(selected_indices)}
- Selection rate: {len(selected_indices)/len(all_frames)*100:.1f}%
- Average quality score: {avg_score:.2f}/10
- Criteria: {selection_criteria}
- Min score threshold: {min_score}
"""
            
            print(f"[QwenFrameSelector] " + "="*60)
            print(f"[QwenFrameSelector] Selection complete!")
            print(summary)
            print(f"[QwenFrameSelector] " + "="*60)
            
            return {
                "ui": {"text": [summary]},
                "result": (frames_tensor, scores_json, summary)
            }
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[QwenFrameSelector] {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                "ui": {"text": [error_msg]},
                "result": (torch.zeros((1, 64, 64, 3)), "{}", error_msg)
            }


NODE_CLASS_MAPPINGS = {
    "QwenFrameSelector": QwenFrameSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenFrameSelector": "Qwen Frame Selector",
}
