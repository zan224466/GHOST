:: Create necessary directories if they don't exist
mkdir arcface_model
mkdir insightface_func\models\antelope
mkdir weights
mkdir AdaptiveWingLoss\AWL_detector
:: Load arcface
curl -L -o ./arcface_model/backbone.pth https://github.com/sberbank-ai/sber-swap/releases/download/arcface/backbone.pth
curl -L -o ./arcface_model/iresnet.py https://github.com/sberbank-ai/sber-swap/releases/download/arcface/iresnet.py

:: Load landmarks detector
curl -L -o ./insightface_func/models/antelope/glintr100.onnx https://github.com/sberbank-ai/sber-swap/releases/download/antelope/glintr100.onnx
curl -L -o ./insightface_func/models/antelope/scrfd_10g_bnkps.onnx https://github.com/sberbank-ai/sber-swap/releases/download/antelope/scrfd_10g_bnkps.onnx

:: Load G and D models with 1, 2, 3 blocks
:: Model with 2 blocks is main
curl -L -o ./weights/G_unet_2blocks.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_2blocks.pth
curl -L -o ./weights/D_unet_2blocks.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_2blocks.pth

curl -L -o ./weights/G_unet_1block.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_1block.pth
curl -L -o ./weights/D_unet_1block.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_1block.pth

curl -L -o ./weights/G_unet_3blocks.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_3blocks.pth
curl -L -o ./weights/D_unet_3blocks.pth https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_3blocks.pth

:: Load model for eyes loss
curl -L -o ./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth https://github.com/sberbank-ai/sber-swap/releases/download/awl_detector/WFLW_4HG.pth

:: Load super res model
curl -L -o ./weights/10_net_G.pth https://github.com/sberbank-ai/sber-swap/releases/download/super-res/10_net_G.pth
