MODELS=(
    "mock"

    "Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    # "bailian-qwen2.5-vl-32b-instruct"
    "api-qwen2.5-vl-72b-instruct"

    "api-claude35_sonnet"
    "api-claude37_sonnet"
    "api-claude_sonnet4"
    "api-claude_opus4"

    "api-gemini-2.0-flash"
    "api-gemini-2.5-pro-06-17"
    "api-gemini-2.5-flash-06-17"

    "api-gpt-4-vision-preview"
    "api-gpt-4o-mini-0718"
    "api-gpt-4o-0806"

    "api-o3-0416-global"
    "api-o4-mini-0416-global"
)

for model in "${MODELS[@]}"; do
    echo "Running for model: $model"
    filename="${model}_$(date +%Y%m%d_%H%M%S).out"
    python -u -m deepphy.runner.05_angrybirds_runner --model "$model" 2>&1 | tee "angrybirds_out/$filename"
done

echo "All models have been processed."
