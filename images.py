import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load your two cropped images
img_a = mpimg.imread('europe_white.png')  # Change to your actual file name
img_b = mpimg.imread('barcelona_white.png') # Change to your actual file name

# Create a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot first image
axes[0].imshow(img_a)
axes[0].axis('off') # Hide axes/grid lines
axes[0].text(-0.05, 1.05, '(a)', transform=axes[0].transAxes,
             fontsize=18, fontweight='bold', va='top')

# Plot second image
axes[1].imshow(img_b)
axes[1].axis('off') # Hide axes/grid lines
axes[1].text(-0.05, 1.05, '(b)', transform=axes[1].transAxes,
             fontsize=18, fontweight='bold', va='top')

# Remove whitespace and save
fig.tight_layout()
plt.savefig('combined_thesis_image.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Images successfully merged and saved!")