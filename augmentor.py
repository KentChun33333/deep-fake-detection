from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, 
    RandomRotate90,
    RandomGamma, RGBShift, 
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)




def augment_flips_color(p=.5):
    return Compose([
        CLAHE(),
        RandomRotate90(),
        RandomGamma(), 
        RGBShift(), 
        Blur(blur_limit=3),
        HueSaturationValue(), 

        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),

        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),

    ], p=p)

# Gray Image no Hue 
def augment_flips_color(p=.5):
    return Compose([
        CLAHE(),
        RandomRotate90(),
        ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value =0), 
        RandomGamma(gamma_limit=(75, 140)), 
        Blur(blur_limit=3),
        # HueSaturationValue(), 

        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),

        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),

    ], p=p)

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

def augment_flips_gray(p=.5):
    '''
    usage:
     au = augment_flips()
     res = au(image=image, mask=make)
     res_img, res_mask = res['image'], res['mask']
    '''
    return Compose([
        # CLAHE(),
        OneOf([
            RandomRotate90(),
            ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value =0)], p=0.35), 

        RandomGamma(gamma_limit=(75, 140)), 
        Blur(blur_limit=3),
        # HueSaturationValue(), 

        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),

        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),

    ], p=p)