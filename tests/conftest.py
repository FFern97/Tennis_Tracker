"""
Hooks de pytest (sin precargar módulos bajo --cov: rompe la medición si se importan
antes de que pytest-cov instale el tracer).
"""
