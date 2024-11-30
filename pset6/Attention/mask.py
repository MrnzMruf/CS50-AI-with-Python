import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"
K = 3  # Number of predictions to generate
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()

    # Print and replace mask token with predictions
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]), result.attentions)

def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    input_ids = inputs.input_ids[0].numpy()  # Get input IDs as a numpy array
    mask_token_indices = (input_ids == mask_token_id).nonzero()  # Find where mask token is located
    return mask_token_indices[0][0].item() if mask_token_indices[0].size > 0 else None  # Return index or None

def get_color_for_attention_score(attention_score):
    """
    Map attention score to a shade of gray RGB tuple.
    """
    gray_value = int(round(attention_score.numpy() * 255))
    return (gray_value, gray_value, gray_value)  # Return gray scale tuple

def visualize_attentions(tokens, attentions):
    """
    Produce diagrams of self-attention scores for each layer and attention head.
    """
    for i, layer in enumerate(attentions):
        for k in range(len(layer[0])):
            layer_number = i + 1
            head_number = k + 1
            generate_diagram(layer_number, head_number, tokens, layer[0][k])

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Create diagrams visualizing self-attention scores.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    for i, token in enumerate(tokens):
        draw.text((PIXELS_PER_WORD // 2, PIXELS_PER_WORD + i * GRID_SIZE), token, fill="white", font=FONT)

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            y = PIXELS_PER_WORD + i * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

if __name__ == "__main__":
    main()
