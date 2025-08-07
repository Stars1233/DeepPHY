set -e

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

PROMPT_FORMATS=(
    "VLA"
    "WM"
)

ANNOTATION_MODES=(
    "unan"
    "an"
)

LEVELS=(
    "s"
    "m"
    "l"
)

# Create output directory if it doesn't exist
mkdir -p kinetix_out

for model in "${MODELS[@]}"; do
    echo "================================================="
    echo "Running for model: $model"
    echo "================================================="
    for prompt_format in "${PROMPT_FORMATS[@]}"; do
        echo "  --> Using prompt format: $prompt_format"
        for annotation_mode in "${ANNOTATION_MODES[@]}"; do
            echo "    --> Using annotation mode: $annotation_mode"
            for level in "${LEVELS[@]}"; do
                echo "      --> Using level: $level"

                filename="${model}_${prompt_format}_${annotation_mode}_${level}_$(date +%Y%m%d_%H%M%S).out"
                CMD="python -u -m deepphy.runner.03_kinetix_runner \"$level\" --model \"$model\" --format \"$prompt_format\" --annotated \"$annotation_mode\""

                echo "        Executing: $CMD"
                echo "        Logging to: kinetix_out/$filename"
                $CMD 2>&1 | tee "kinetix_out/$filename"

                echo "      --> Finished level: $level"
            done
        done
    done
done

echo "All evaluation runs are complete."
