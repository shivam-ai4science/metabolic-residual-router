from src.models.span_guided_dpi import SpanGuidedDPI
from src.models.two_tower_mlp import BilinearTwoTowerMLP
from src.models.dpi_bi_directional import BiDirectionalCrossAttentionDPI
from src.models.dpi_bi_ligand_pooled import BiDirectionalCrossAttentionDPILigandPooled
from src.models.dpi_uni_ligand_pooled import UnidirectionalCrossAttentionLigandPooled
from src.models.coldstart_hybrid import ColdstartHybridDPI
from src.models.coldstart_cpi_repro import ColdstartCPIRepro
from src.models.explainable_coldstart_cpi import ExplainableColdstartCPI
from src.models.contactformer_cpi import ContactFormerCPI
from src.models.contactformer_v2_cpi import ContactFormerV2CPI
from src.models.pathway_gated_coldstart_cpi import PathwayGatedColdstartCPI
from src.models.routed_metabolic_residual_cpi import RoutedMetabolicResidualCPI

def get_model(config_dict):
    
    model_name = config_dict['model']['name']
    model_params = config_dict['model']['params']
    
    models = {
        "SpanGuidedDPI": SpanGuidedDPI,
        "BilinearTwoTowerMLP": BilinearTwoTowerMLP,
        "BiDirectionalCrossAttentionDPI": BiDirectionalCrossAttentionDPI,
        "BiDirectionalCrossAttentionDPILigandPooled": BiDirectionalCrossAttentionDPILigandPooled,
        "UnidirectionalCrossAttentionLigandPooled": UnidirectionalCrossAttentionLigandPooled,
        "ColdstartHybridDPI": ColdstartHybridDPI,
        "ColdstartCPIRepro": ColdstartCPIRepro,
        "ExplainableColdstartCPI": ExplainableColdstartCPI,
        "ContactFormerCPI": ContactFormerCPI,
        "ContactFormerV2CPI": ContactFormerV2CPI,
        "PathwayGatedColdstartCPI": PathwayGatedColdstartCPI,
        "RoutedMetabolicResidualCPI": RoutedMetabolicResidualCPI,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found! Available: {list(models.keys())}")

    return models[model_name](**model_params)
