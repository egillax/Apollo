class DataNames:
    """
    Names of the various fields in the input data as generated by cehr_bert_cdm_processor
    """
    CONCEPT_IDS = "concept_ids"
    VISIT_SEGMENTS = "visit_segments"
    DATES = "dates"
    AGES = "ages"
    VISIT_CONCEPT_ORDERS = "visit_concept_orders"
    VISIT_CONCEPT_IDS = "visit_concept_ids"
    ORDERS = "orders"
    NUM_OF_CONCEPTS = "num_of_concepts"
    NUM_OF_VISITS = "num_of_visits"
    LABEL = "label"
    SEQUENCE_LENGTH = "num_of_concepts"


class ModelInputNames:
    """
    Names of the inputs to the model. These inputs are generated by the data generator using the learning objectives.
    """
    TOKEN_IDS = "token_ids"
    PADDING_MASK = "padding_mask"
    MASKED_TOKEN_IDS = "masked_token_ids"
    MASKED_TOKEN_MASK = "masked_token_mask"
    DATES = "dates"
    VISIT_SEGMENTS = "visit_segments"
    AGES = "ages"
    VISIT_CONCEPT_ORDERS = "visit_concept_orders"
    VISIT_TOKEN_IDS = "visit_token_ids"
    MASKED_VISIT_TOKEN_IDS = "masked_visit_token_ids"
    MASKED_VISIT_TOKEN_MASK = "mask_visit_token_mask"
    NEXT_TOKEN_IDS = "next_token_ids"
    NEXT_VISIT_TOKEN_SET = "next_visit_token_set"
    FINETUNE_LABEL = "finetune_label"
    
    
class ModelOutputNames:
    """
    Names of the outputs to the model.
    """
    TOKEN_PREDICTIONS = "token_predictions"
    VISIT_TOKEN_PREDICTIONS = "visit_token_predictions"
    NEXT_TOKEN_PREDICTION = "next_token_prediction"
    NEXT_VISIT_TOKENS_PREDICTION = "next_visit_tokens_prediction"
    LABEL_PREDICTIONS = "label_predictions"
