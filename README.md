# Repository of the Thesis: Evaluating Computer Vision Generative Models for Synthetic Data Generation in Industry

## Data Preparation

1. To prepare the dataset, utilize the scripts located in `master_thesis/data_preparation`.
2. To generate additional images from a single instance, run `master_thesis/data_preparation/augment_data.py`.
3. To impair (superimpose) data and label it, use `master_thesis/data_preparation/expand_labelled_data.py`.
4. To convert images to a square format, execute `master_thesis/data_preparation/make_image_square.py`.
5. Mask processing can be performed using `master_thesis/data_preparation/mask_preprocessing.py`.
6. Generated images can be superimposed by running `master_thesis/data_preparation/super_impose_images.py`.
7. An explanatory guide is available in the Jupyter Notebooks.

## Modeling

1. To utilize ControlNet, follow the path `master_thesis/models/diffusion_models/controlnet`.
2. For Stable Diffusion and its various versions, navigate to `C:\Users\kaleem\Learn\Wuerth\master_thesis\models\diffusion_models\stable_diffusion2`.
3. To use Kandinsky and its variants, refer to `master_thesis/models/diffusion_models/Kandinsky`.
4. For Instruct Pix2Pix, follow the path `master_thesis/models/diffusion_models/instruct_pix2pix`.
5. The MLGIE repository has been cloned for use with specific supportive modules.

## Evaluation

1. Evaluation processes are located in `master_thesis/evaluation`.
2. To calculate evaluation metrics, utilize `master_thesis/evaluation/evaluation_metrics.py`.
3. Jupyter Notebooks provide additional guidance and evaluation models.

Feel free to explore each section for detailed instructions and resources!
