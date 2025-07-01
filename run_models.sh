#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# If you prefer to create a brand-new test CSV from an old predictions CSV each time, set:
OLD_PRED_CSV="results/inception_v3_pretrained/predictions.csv"
# or if you already have a test.csv, comment out the above and set:
# TEST_CSV="test.csv"

# 1) ResNet-50
python src/test_inference_with_metrics.py \
    --old_pred_csv "$OLD_PRED_CSV" \
    --generated_test_csv "output/resnet50_test.csv" \
    --model_type resnet50 \
    --weights weights/resnet50_granule.pth \
    --device gpu \
    --batch_size 16 \
    --output_csv output/resnet50_predictions.csv \
    --plot_file output/resnet50_plot.png \
    > output/resnet50_output.txt 2>&1

# 2) EfficientNet-B0
python src/test_inference_with_metrics.py \
    --old_pred_csv "$OLD_PRED_CSV" \
    --generated_test_csv "output/efficientnet_b0_test.csv" \
    --model_type efficientnet_b0 \
    --weights weights/efficientnet_b0_best.pth \
    --device gpu \
    --batch_size 16 \
    --output_csv output/efficientnet_b0_predictions.csv \
    --plot_file output/efficientnet_b0_plot.png \
    > output/efficientnet_b0_output.txt 2>&1

# 3) Inception v3
python src/test_inference_with_metrics.py \
    --old_pred_csv "$OLD_PRED_CSV" \
    --generated_test_csv "output/inception_v3_test.csv" \
    --model_type inception_v3 \
    --weights weights/inception_v3_best.pth \
    --device gpu \
    --batch_size 16 \
    --output_csv output/inception_v3_predictions.csv \
    --plot_file output/inception_v3_plot.png \
    > output/inception_v3_output.txt 2>&1

echo "All models have finished. See the 'output/' directory for results."
