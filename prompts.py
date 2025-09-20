# prompts.py

NATURALIZE_PROMPT = """Tu es un rédacteur professionnel.
Tu reçois un transcript brut, haché par segments de sous-titres (coupures en plein milieu, hésitations, erreurs).
Tâches :
1) Fusionner en phrases complètes et fluides.
2) Corriger orthographe, grammaire et ponctuation.
3) Supprimer les hésitations et répétitions inutiles.
4) Préserver strictement le sens et le ton.
5) Unifier le style sur ce morceau.
Réponds uniquement par le texte final, sans ajouter de commentaires.
Texte :
---
{chunk}
---
"""

TRANSLATE_PROMPT = """Tu es traducteur professionnel.
Traduis le texte suivant en {target_lang} avec un style naturel et fluide, en préservant strictement le sens et le ton.
Corrige au passage les petites erreurs et coupe les répétitions.
Ne commente pas la sortie, réponds uniquement par le texte traduit.
Texte :
---
{chunk}
---
"""
