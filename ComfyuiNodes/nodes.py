
from .DPRW_utils import *
from .CustomNodes import DPRExtractor,DPRKSamplerAdvanced
from .GaussianShading import GSWatermark,GSLatent
from .DPRW_Watermark import DPRWatermark,DPRLatent
from .TreeRing import TreeRingLatent,TreeRingWatermarker



NODE_CLASS_MAPPINGS = {
    "DPR_Latent": DPRLatent,
    "DPR_Extractor": DPRExtractor,
    "DPR_KSamplerAdvanced": DPRKSamplerAdvanced,
    "DPR_GS_Latent": GSLatent,
    "DPRW_TreeRingLatent": TreeRingLatent,
    "DPR_ImageQualityMetrics": ImageQualityMetrics,
    "DPR_DifferenceGenerator": DifferenceGenerator,
    "DPR_AttackSimulator": AttackSimulator,
    "DPR_MetadataGenerator": DPRMetadataGenerator,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DPR_Latent": "DPR Latent",
    "DPR_Extractor": "DPR Extractor",
    "DPR_KSamplerAdvanced": "DPR KSampler Advanced",
    "DPR_GS_Latent": "DPR GS Latent Noise",
    "DPRW_TreeRingLatent":"TreeRing Latent",
    "DPR_ImageQualityMetrics": "DPR Image Quality Metrics",
    "DPR_DifferenceGenerator": "DPR Difference Generator",
    "DPR_AttackSimulator": "DPR Attack Simulator",
    "DPR_MetadataGenerator": "DPR Metadata Generator",
}