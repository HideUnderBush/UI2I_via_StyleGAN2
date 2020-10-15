# 19, 11
# i 9 d 5 pose
# i 9 d 7 age (not good for anime) 
python3 apply_factor.py -i 12 -d 5 -n 10 --ckpt checkpoint/120000.pt factors/factor_anime_finetune.pt --size 256 --out_prefix finetune
python3 apply_factor.py -i 12 -d 5 -n 10 --ckpt checkpoint/anime_freezeFC_G1-4_22000.pt factors/factor_anime_FC.pt --size 256 --out_prefix FC
python3 apply_factor.py -i 12 -d 5 -n 10 --ckpt checkpoint/network-ffhq-256.pt factors/factor_face_256.pt --size 256 --out_prefix face

