test_collection=$1
text_feat_name=$2
video_feat_name=$3
gt_file_name=$4

python evaluation.py --local_rank=0 \
    --text_feat_path="/home/nhattanpro444/Project/TeachCLIP/reproduce_TeachCLIP/data/datasets/$test_collection/FeatureData/$text_feat_name/xclip/xclip/xclip" \
    --video_feat_path="/home/nhattanpro444/Project/TeachCLIP/reproduce_TeachCLIP/data/datasets/$test_collection/FeatureData/$video_feat_name/xclip/xclip/xclip" \
    --gt_file_path="/home/nhattanpro444/Project/TeachCLIP/reproduce_TeachCLIP/data/datasets/msrvtt/Annotations/$gt_file_name.txt"