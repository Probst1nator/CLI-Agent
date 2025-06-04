# import os
# import torch
# from PIL import Image # For saving masks as images
# from typing import Optional, Any, Dict, Tuple, List
# import importlib
# import warnings
# # numpy, matplotlib.pyplot, cv2 will be imported lazily

# # --- Minimal Base Class (if py_classes.cls_util_base is not available) ---
# try:
#     from py_classes.cls_util_base import UtilBase
# except ImportError:
#     class UtilBase:
#         pass
#     warnings.warn("Could not import UtilBase from py_classes. Using a dummy UtilBase. "
#                   "Ensure 'py_classes/cls_util_base.py' is correctly set up.", UserWarning, stacklevel=2)

# class SegmentImage(UtilBase):
#     _loaded_models: Dict[Tuple[str, torch.device], Any] = {}
#     _sam_imports = None

#     @classmethod
#     def _ensure_dependencies(cls) -> bool:
#         if cls._sam_imports is None:
#             try:
#                 matplotlib_module = importlib.import_module('matplotlib')
#                 # Set Matplotlib backend to 'Agg' BEFORE importing pyplot to avoid GUI issues
#                 # if no other backend is specified via environment variable.
#                 if os.environ.get('MPLBACKEND') is None:
#                     current_backend = matplotlib_module.get_backend()
#                     if not current_backend.lower() == 'agg':
#                          matplotlib_module.use('Agg')
                
#                 sam_model_registry = importlib.import_module('segment_anything').sam_model_registry
#                 SamAutomaticMaskGenerator = importlib.import_module('segment_anything').SamAutomaticMaskGenerator
#                 cv2_lib = importlib.import_module('cv2')
#                 np_lib = importlib.import_module('numpy')
#                 plt_lib = importlib.import_module('matplotlib.pyplot')
                
#                 cls._sam_imports = {
#                     'sam_model_registry': sam_model_registry,
#                     'SamAutomaticMaskGenerator': SamAutomaticMaskGenerator,
#                     'cv2': cv2_lib,
#                     'np': np_lib,
#                     'plt': plt_lib,
#                 }
#                 return True
#             except ImportError as e:
#                 error_msg = (
#                     f"Required libraries for SegmentImage not found: {e}. "
#                     "Please install them: pip install torch torchvision segment-anything opencv-python numpy matplotlib pillow"
#                 )
#                 warnings.warn(error_msg, UserWarning, stacklevel=2)
#                 cls._sam_imports = None
#                 return False
#         return True

#     @classmethod
#     def _get_sam_model(cls, model_type: str, checkpoint_path: str, device: torch.device) -> Any:
#         if not cls._ensure_dependencies():
#             raise ImportError("Required dependencies not available.")
            
#         cache_key = (model_type, device)
#         if cache_key in cls._loaded_models:
#             print(f"Using cached SAM model '{model_type}' on {device}")
#             return cls._loaded_models[cache_key]
            
#         print(f"Loading SAM model '{model_type}' from '{checkpoint_path}' to {device}...")
#         sam_model_registry = cls._sam_imports['sam_model_registry']
#         try:
#             sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
#             sam.to(device=device)
#             sam.eval()
#         except Exception as e:
#             raise IOError(f"Failed to load SAM model from '{checkpoint_path}'. "
#                           f"Ensure path is correct, file is not corrupted, and matches model_type '{model_type}'. Error: {e}")
        
#         print(f"SAM model '{model_type}' loaded successfully.")
#         cls._loaded_models[cache_key] = sam
#         return sam

#     @classmethod
#     def unload_sam_model(cls, model_type: str = None, device: torch.device = None) -> int:
#         keys_to_remove = []
#         for key in list(cls._loaded_models.keys()):
#             key_model_type, key_device = key
#             if (model_type is None or key_model_type == model_type) and \
#                (device is None or key_device == device):
#                 keys_to_remove.append(key)
        
#         for key in keys_to_remove:
#             model = cls._loaded_models.pop(key)
#             if hasattr(model, 'to') and hasattr(model, 'device') and str(model.device) != 'cpu':
#                 try: model.to('cpu')
#                 except Exception as e: warnings.warn(f"Failed to move model {key} to CPU: {e}", UserWarning, stacklevel=2)
#             del model
            
#         import gc
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         return len(keys_to_remove)

#     @staticmethod
#     def _get_device() -> torch.device:
#         if torch.cuda.is_available(): return torch.device("cuda")
#         if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
#             try:
#                 x = torch.tensor([1.0], device="mps"); _ = x * x; return torch.device("mps")
#             except RuntimeError: warnings.warn("MPS available but test failed; using CPU.", UserWarning, stacklevel=2)
#         return torch.device("cpu")

#     @staticmethod
#     def _show_anns(anns: List[Dict[str, Any]]):
#         # This static method directly uses SegmentImage._sam_imports if already populated by _ensure_dependencies
#         if not SegmentImage._sam_imports: # Ensure dependencies are loaded if called standalone (less likely)
#             if not SegmentImage._ensure_dependencies(): return
        
#         plt = SegmentImage._sam_imports['plt']
#         np = SegmentImage._sam_imports['np']

#         if not anns: return
#         sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#         ax = plt.gca()
#         ax.set_autoscale_on(False)
        
#         h, w = sorted_anns[0]['segmentation'].shape
#         img_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
#         for ann in sorted_anns:
#             m = ann['segmentation']
#             color_mask = np.random.randint(0, 256, 3, dtype=np.uint8)
#             img_overlay[m] = np.concatenate([color_mask, [int(0.35 * 255)]])
#         ax.imshow(img_overlay)

#     @classmethod
#     def run(
#         cls,
#         image_path: str,
#         output_dir: str,
#         model_type: str = "vit_b", # Default to ViT-Base
#         # Checkpoint path construction assumes 'assets' is sibling to script's parent dir
#         checkpoint_fname: str = "sam_vit_b_01ec64.pth",
#         save_overlay_image: bool = True,
#         save_individual_masks: bool = False,
#         display_results: bool = False, # Note: plt.show() might not work with 'Agg' backend
#         automatic_mask_generator_params: Optional[Dict[str, Any]] = None
#     ) -> List[str]:
#         if not cls._ensure_dependencies():
#             raise RuntimeError("Segment Anything dependencies are not met.")

#         # Construct full checkpoint path
#         # Assumes script is in utils/, assets/ is in parent of utils/
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(script_dir)
#         checkpoint_path = os.path.join(project_root, "assets", checkpoint_fname)
        
#         cv2 = cls._sam_imports['cv2']
#         np = cls._sam_imports['np']
#         plt = cls._sam_imports['plt']
#         SamAutomaticMaskGenerator = cls._sam_imports['SamAutomaticMaskGenerator']

#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Input image not found: {image_path}")
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}. "
#                                     f"Ensure '{checkpoint_fname}' is in '{os.path.join(project_root, 'assets')}'")

#         os.makedirs(output_dir, exist_ok=True)
#         saved_file_paths: List[str] = []
#         base_filename = os.path.splitext(os.path.basename(image_path))[0]

#         device = cls._get_device()
#         print(f"Using device: {device}")

#         try:
#             sam = cls._get_sam_model(model_type, checkpoint_path, device)
#             image = cv2.imread(image_path)
#             if image is None: raise IOError(f"Could not load image: {image_path}")
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             print("Performing automatic mask generation...")
#             mask_gen_kwargs = automatic_mask_generator_params or {}
#             mask_generator = SamAutomaticMaskGenerator(sam, **mask_gen_kwargs)
#             masks = mask_generator.generate(image_rgb)
#             print(f"Found {len(masks)} masks.")

#             if not masks:
#                 print("No masks generated."); return []

#             if save_overlay_image:
#                 dpi = 100
#                 fig = plt.figure(figsize=(image_rgb.shape[1]/dpi, image_rgb.shape[0]/dpi), dpi=dpi)
#                 plt.imshow(image_rgb)
#                 cls._show_anns(masks)
#                 plt.title(f"Segmentation: {os.path.basename(image_path)}")
#                 plt.axis('off')
#                 overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")
#                 fig.canvas.draw() # Important for 'Agg' backend
#                 plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
#                 saved_file_paths.append(os.path.abspath(overlay_path))
#                 print(f"Saved overlay to: {overlay_path}")
#                 if display_results:
#                     warnings.warn("display_results=True ineffective with 'Agg' backend.", UserWarning, stacklevel=2)
#                     # plt.show() # plt.show() does nothing with 'Agg' backend
#                 plt.close(fig)

#             if save_individual_masks:
#                 masks_dir = os.path.join(output_dir, f"{base_filename}_masks")
#                 os.makedirs(masks_dir, exist_ok=True)
#                 for i, ann in enumerate(masks):
#                     mask_img = Image.fromarray(ann['segmentation'].astype(np.uint8) * 255, 'L')
#                     mask_fpath = os.path.join(masks_dir, f"mask_{i+1:04d}.png")
#                     mask_img.save(mask_fpath)
#                     saved_file_paths.append(os.path.abspath(mask_fpath))
#                 print(f"Saved {len(masks)} individual masks to: {masks_dir}")
#         except Exception as e:
#             print(f"Error during segmentation: {e}")
#             import traceback; traceback.print_exc()
#             raise RuntimeError(f"Segmentation failed: {e}") from e
#         return saved_file_paths

# # --- Main Test Block ---
# if __name__ == "__main__":
#     print("--- Testing SegmentImage ---")
#     temp_output_dir = "temp_segmented_output_sam"
#     if not os.path.exists(temp_output_dir):
#         os.makedirs(temp_output_dir)

#     # Define paths relative to this script's location
#     # Assumes script is in utils/, assets/ is in parent of utils/, test_image.png is in assets/
#     current_script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root_dir = os.path.dirname(current_script_dir)
    
#     test_image_file = os.path.join(project_root_dir, "assets", "test_image.png")
#     # For ViT-Base model, this is the expected checkpoint name
#     sam_checkpoint_filename = "sam_vit_b_01ec64.pth" 
#     sam_checkpoint_full_path = os.path.join(project_root_dir, "assets", sam_checkpoint_filename)

#     # Check for essential files
#     if not os.path.exists(test_image_file):
#         print(f"ERROR: Test image '{test_image_file}' not found. Please create it or provide a valid path.")
#         exit(1)
    
#     if not os.path.exists(sam_checkpoint_full_path):
#         print(f"WARNING: SAM checkpoint '{sam_checkpoint_full_path}' not found.")
#         print(f"Please download '{sam_checkpoint_filename}' and place it in '{os.path.join(project_root_dir, 'assets')}' for the test to run correctly.")
#         print("Segmentation will likely fail without the correct checkpoint.")
#         # We can still try to run to test other parts, but model loading will fail.
#     else:
#         print(f"Using checkpoint: {sam_checkpoint_full_path}")

#     test_device = SegmentImage._get_device()
#     print(f"Test will run on device: {test_device}")

#     try:
#         print("\n--- Running Automatic Segmentation Test ---")
#         # Call run with explicit model_type and checkpoint_fname for clarity in test
#         # These match the defaults in the run method signature but are explicit here.
#         output_files = SegmentImage.run(
#             image_path=test_image_file,
#             output_dir=temp_output_dir,
#             model_type="vit_b", 
#             checkpoint_fname=sam_checkpoint_filename,
#             save_overlay_image=True,
#             save_individual_masks=True # Test this functionality too
#         )
        
#         print(f"\nSegmentation process completed. Output files: {output_files}")

#         if os.path.exists(sam_checkpoint_full_path) and os.path.getsize(sam_checkpoint_full_path) > 1000:
#             assert len(output_files) > 0, "Expected output files from segmentation."
#             print("Output file(s) generated.")
#             # Further check if the main overlay exists
#             expected_overlay = os.path.join(temp_output_dir, f"{os.path.splitext(os.path.basename(test_image_file))[0]}_overlay.png")
#             assert os.path.exists(expected_overlay), f"Expected overlay file '{expected_overlay}' not found."
#             print(f"Overlay file '{expected_overlay}' confirmed.")
#             if True: # If save_individual_masks was True
#                 expected_mask_dir = os.path.join(temp_output_dir, f"{os.path.splitext(os.path.basename(test_image_file))[0]}_masks")
#                 assert os.path.isdir(expected_mask_dir), f"Expected individual masks directory '{expected_mask_dir}' not found."
#                 print(f"Individual masks directory '{expected_mask_dir}' confirmed.")
#         else:
#             print("Skipping detailed output file assertions due to missing/dummy checkpoint.")

#         print("\n--- Testing Model Unloading ---")
#         unloaded_count = SegmentImage.unload_sam_model(model_type="vit_b", device=test_device)
#         print(f"Unloaded {unloaded_count} SAM model(s) of type 'vit_b' on device '{test_device}'.")
#         assert SegmentImage._loaded_models.get(("vit_b", test_device)) is None, "Model 'vit_b' was not properly unloaded."
#         if os.path.exists(sam_checkpoint_full_path) and os.path.getsize(sam_checkpoint_full_path) > 1000 and output_files:
#              assert unloaded_count > 0, "Expected at least one model to be unloaded."
        
#         print("\nAll tests passed (or skipped gracefully where dependencies were missing).")

#     except FileNotFoundError as fnf_err:
#         print(f"TEST FAILED (FileNotFoundError): {fnf_err}")
#     except ImportError as imp_err:
#         print(f"TEST FAILED (ImportError): {imp_err}. Check installations.")
#     except RuntimeError as rt_err:
#         print(f"TEST FAILED (RuntimeError): {rt_err}")
#     except Exception as e:
#         print(f"TEST FAILED (Unexpected Error): {type(e).__name__} - {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n--- Finished testing SegmentImage ---")