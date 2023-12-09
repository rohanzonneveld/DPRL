from PIL import Image
import os

# Open the PNG files
print(os.getcwd())
os.chdir('assignment3/figures')
image_files = ['starting_state.png', 'state_post_move1.png', 'state_post_move2.png', 'state_post_move3.png', 'state_post_move4.png', 'state_post_move5.png', 'state_post_move6.png', 'state_post_move7.png']
images = [Image.open(filename) for filename in image_files]

# Assuming all images are the same size, calculate the size of the final grid
num_cols = 4
num_rows = 2
images_per_row = 4
image_width, image_height = images[0].size
grid_width = images_per_row * image_width
grid_height = num_rows * image_height

# Create a new blank image with the calculated size
combined_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))  # Adjust 'RGB' based on your image mode

# Paste each image onto the new blank image in a grid pattern
for i, im in enumerate(images):
    row = i // images_per_row
    col = i % images_per_row
    x_offset = col * image_width
    y_offset = row * image_height
    combined_image.paste(im, (x_offset, y_offset))

# Save the combined image
combined_image.save('combined_image_grid.png')