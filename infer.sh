CUDA_VISIBLE_DEVICES="4" python cityscapes_infer.py \
--ckpt="/DATA_EDS2/AIGC/2312/xuhr2312/workspace/ControlNet/work_dir/cityscapes_hr/ckpt_hr/epoch=45-step=19999.ckpt" \
--images=1 \
--save_path='temp/inference_img'  \
--model='models/cldm_v21.yaml'