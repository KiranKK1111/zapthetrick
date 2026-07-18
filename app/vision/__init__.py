"""Local Vision Intelligence Layer (VisionAnalysis.md).

A local vision model is the universal image reader: it parses screenshots /
documents / photos into structured TEXT once (cached per image), which is then
handed to any provider TEXT model. No API/provider VISION model is ever used.

Public entry point: `from app.vision import factory; await
factory.describe_images(images_b64, prompt)`.
"""
