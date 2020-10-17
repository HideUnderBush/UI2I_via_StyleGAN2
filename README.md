We proposed an unsupervised image-to-image translation method via pre-trained StyleGAN2 network. 

paper: [](arxiv link)

### Step 1: Model Fine-tuning
Follow the instruction stated in the StyleGAN2 pytorch implementation [](here) (we modify the train.py code to freeze FC)
(obtain the target model)

### Step 2: Closed-Form GAN space
Calculate the GAN space via the proposed algorithm
(obtain the factor)
``python3 closed_form_factorization.py --ckpt your_model --out output_factor_path``

### Step 3: Image inversion
Inverse the image to a latent code based on the StyleGAN2 model trained on its domain
``python3 project_factor.py --ckpt stylegan_modle_path --fact factor_path IMAGE_FILE``

### Step 4: Image generation with multiple styles
We use the inversed code to generate images with multiple style in the target domain

``python3 gen_multi_style.py --model base_model_path --model2 target_model_path  --fact base_inverse.pt --fact_base factor_from_base_model -o output_path --swap_layer 3 --stylenum 10``

In additon to multi-modal translation, the style of the output can be specified by reference. To achieve this, we need to inverse the reference image as well since its latent code would then be used as style code in the generation. 

``python3 gen_ref.py --model1 base_model_path --model2 target_model_path --fact base_inverse.pt --fac_ref reference_inverse.pt --fact_base1 factor_from_base_model --fact_base2 factor_from_target_model -o output_path``

### pre-trained base model and dataset
We use the StyleGAN2 face models trained on FFHQ, [](256x256) and [](1024x1024), models fine-tuned on such models can be used for I2I translation, though with FreezeFC, it can achieve better results. 
Many thanks to [](Gwern) for providing the Anime dataset and [](D&J) for providing the cartoon dataset. 

### Some Results
(teaser + multi-domain results?)
1. anime2face?
2. cartoon2face?
3. multi-modal & with ref
4. multi-domain A2B2C (triangle?)

The code is heavily borrowed from StyleGAN2 implementation ([rosality](pytorch version)) and [close-form](closed-form GAN space), thanks to their great work and contribution!

