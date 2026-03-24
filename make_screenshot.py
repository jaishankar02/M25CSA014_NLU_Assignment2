import sys
from PIL import Image, ImageDraw, ImageFont

def create_terminal_screenshot(text_file, output_image):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Determine image size based on text
    font_size = 14
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate max width and total height
    # Since we might not have truetype, we estimate for default font
    char_width = 8
    char_height = 16
    
    max_line_len = max([len(line.rstrip('\n')) for line in lines] + [80])
    img_width = max(max_line_len * char_width + 40, 800)
    img_height = max(len(lines) * char_height + 40, 600)

    # Create image with dark background
    img = Image.new('RGB', (img_width, img_height), color='#1e1e1e')
    draw = ImageDraw.Draw(img)

    # Draw text
    y_text = 20
    for line in lines:
        draw.text((20, y_text), line.rstrip('\n'), font=font, fill='#00ff00')
        y_text += char_height

    img.save(output_image)
    print(f"Saved {output_image}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_screenshot.py <input.txt> <output.png>")
        sys.exit(1)
    create_terminal_screenshot(sys.argv[1], sys.argv[2])
