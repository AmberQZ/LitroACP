# LitroACP: Lightweight & Robust Access Control Policy Extraction Framework

Official implementation for the paper ​【LitroACP: A Lightweight and Robust Framework for Extracting Access Control Policies from Specifications】(CAiSE'25).

## Overview
LitroACP is a novel framework for automated extraction of fine-grained access control policies from natural language specifications. Our solution combines:
- 🏷️ ​**ACPUIE**(annotated_datasets): Semi-automatic annotation tool for policy component labeling
- 🛡️ ​**DisAdver**(decision_identification): Robust policy decision identification with adversarial training
- 🔍 ​**GLiACP**(component_extraction): Efficient policy component extraction using knowledge-enhanced NER

Key features:

✔️ Lightweight architecture (116M total parameters)  
✔️ State-of-the-art performance (93.77% F1 for NLACP identification)  
✔️ Domain-agnostic policy extraction  
✔️ Comprehensive evaluation on real-world datasets


​**annotated_datasets/data_acp​** is our annotated dataset for access control policy.

​**annotated_datasets/data_non​** is our annotated dataset for non-access control policy(not used in this work but can be used for furthur research).
