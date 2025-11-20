# Check and download sam-ckpt
if [ ! -f ./ckpt/sam_vit_b_01ec64.pth ]; then
    wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "sam_vit_b_01ec64.pth already downloaded."
fi


# Check and download aot-ckpt 
if [ ! -f ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth ]; then
    gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
else
    echo "R50_DeAOTL_PRE_YTB_DAV.pth already downloaded."
fi

# Check and download aot-ckpt 
if [ ! -f ./ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth ]; then
    gdown '1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq' --output ./ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth
else
    echo "SwinB_DeAOTL_PRE_YTB_DAV.pth already downloaded."
fi

# Check and download aot-ckpt 
if [ ! -f ./ckpt/dinov2_vitb14_reg4_pretrain.pth ]; then
    wget -P ./ckpt https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
else
    echo "dinov2_vitb14_reg4_pretrain.pth already downloaded."
fi

# Check and download dinov3-ckpt
if [ ! -f ./ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth ]; then
    gdown '18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6' --output ./ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
else
    echo "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth already downloaded."
fi

# Check and download dinov3-ckpt
if [ ! -f ./ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ]; then
    gdown '195H5UHKJ0r4qRDY7Ly6WJrXGnpdlHMSu' --output ./ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
else
    echo "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth already downloaded."
fi


