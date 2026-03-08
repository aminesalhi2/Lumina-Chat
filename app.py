"""
LUMINA CHAT - Chatbot intelligent BERT + Generation Dynamique
Usage: streamlit run app.py
"""

# ═══════════════════════════════════════════════════════════
# 1. IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════
import re, time, random, unicodedata
from functools import lru_cache
from typing import Optional

import nltk, numpy as np, streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

for _p in ["punkt","stopwords","wordnet","punkt_tab"]:
    try: nltk.download(_p, quiet=True)
    except: pass

CONFIDENCE_THRESHOLD = 0.50
BERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_CACHE_SIZE  = 256
TOP_K           = 4

# ═══════════════════════════════════════════════════════════
# 2. CORPUS MULTIDOMAINE
# ═══════════════════════════════════════════════════════════
CORPUS = {
    "greetings": {
        "patterns": ["bonjour","bonsoir","salut","coucou","hey","hello","hi",
                     "yo","wesh","cc","bjr","bsr","slt","ola","hola"],
        "responses": {
            "hello": [
                "Bonjour ! 😊 Comment puis-je vous aider aujourd'hui ?",
                "Salut ! 👋 Je suis là pour répondre à vos questions.",
                "Hello ! 😄 Qu'est-ce que je peux faire pour vous ?",
                "Bonjour ! ✨ Posez-moi votre question, je suis tout ouïe.",
            ],
            "goodbye": [
                "À bientôt ! 👋 Ce fut un plaisir de vous aider.",
                "Au revoir ! 😊 N'hésitez pas à revenir.",
                "Bonne journée ! ☀️ À la prochaine.",
                "Ciao ! 🌟 Prenez soin de vous.",
            ],
        },
    },
    "emotions": {
        "patterns": ["je suis triste","je suis fatigué","je me sens mal",
                     "je suis stressé","j'ai peur","ça va","comment tu vas"],
        "responses": [
            "Je comprends ce que vous ressentez. 💙 Voulez-vous en parler ?",
            "Merci de partager ça avec moi. 🤗 Comment puis-je vous aider ?",
            "Je suis là pour vous écouter. ✨ Qu'est-ce qui se passe ?",
            "Moi, je fonctionne parfaitement ! 😄 Et vous, comment allez-vous ?",
            "Je suis un programme mais je 'sens' que vous méritez une bonne réponse ! 🤖",
        ],
    },
    "about_bot": {
        "patterns": ["qui es-tu","tu es quoi","lumina","parle-moi de toi",
                     "tes capacités","comment tu fonctionnes","tu es intelligent"],
        "responses": [
            "Je suis Lumina 🌟, un assistant IA basé sur BERT multilingue pour la compréhension sémantique. Je couvre l'économie, l'informatique, la politique, la santé, les sciences et bien plus !",
            "Je suis un chatbot intelligent propulsé par des modèles transformer. 🤖 Je comprends vos questions grâce à BERT et génère des réponses dynamiques et contextuelles !",
            "Lumina, c'est moi ! ✨ Je combine BERT pour comprendre votre intention et une génération contextuelle pour vous répondre. Posez-moi n'importe quelle question !",
        ],
    },
    "economie": [
        {
            "q": "Qu'est-ce que le PIB ?",
            "k": "Le Produit Intérieur Brut (PIB) mesure la valeur totale des biens et services produits dans un pays sur une période donnée. C'est le principal indicateur de la santé économique. Il peut croître (expansion) ou reculer deux trimestres consécutifs (récession). Il est calculé selon trois approches : production, revenus ou dépenses.",
        },
        {
            "q": "Comment fonctionne l'inflation ?",
            "k": "L'inflation est la hausse générale et durable des prix. Elle érode le pouvoir d'achat : avec le même argent, on achète moins. Elle est mesurée par l'Indice des Prix à la Consommation (IPC). Les banques centrales visent 2% par an. Les causes : excès de demande, hausse des coûts de production ou création monétaire excessive.",
        },
        {
            "q": "Qu'est-ce que la bourse et comment investir ?",
            "k": "La bourse est un marché organisé où s'échangent des actions, obligations, ETF et dérivés. Les prix fluctuent selon l'offre, la demande et les conditions macroéconomiques. Pour investir, on utilise un compte-titres ou un PEA en France. La diversification réduit les risques. Les grands indices : CAC 40 (France), Dow Jones et NASDAQ (USA).",
        },
        {
            "q": "Qu'est-ce que les cryptomonnaies ?",
            "k": "Les cryptomonnaies sont des monnaies numériques décentralisées basées sur la cryptographie et la blockchain. Bitcoin, créé en 2009 par Satoshi Nakamoto, fut la première. Ethereum introduit les contrats intelligents. Elles permettent des transactions sans intermédiaire bancaire mais sont très volatiles et soulèvent des questions réglementaires.",
        },
        {
            "q": "Qu'est-ce que la mondialisation économique ?",
            "k": "La mondialisation désigne l'intégration croissante des économies via le commerce, les flux de capitaux et la technologie. Elle a réduit la pauvreté mondiale mais accentué les inégalités dans certains pays développés. Le libre-échange, théorisé par Ricardo (avantage comparatif), en est le moteur principal.",
        },
        {
            "q": "Qu'est-ce que la théorie keynésienne ?",
            "k": "Développée par John Maynard Keynes dans les années 1930, la théorie keynésienne soutient que la demande globale détermine le niveau de production et d'emploi. En récession, l'État doit stimuler l'économie par des dépenses publiques plutôt qu'attendre l'autorégulation des marchés. Elle justifie les plans de relance modernes.",
        },
        {
            "q": "Comment fonctionne la politique monétaire ?",
            "k": "La politique monétaire est conduite par les banques centrales (BCE, Fed) pour contrôler la masse monétaire et les taux d'intérêt. Ses outils : le taux directeur, les opérations d'open market et les réserves obligatoires. En baissant les taux, elle stimule le crédit et l'investissement ; en les relevant, elle freine l'inflation.",
        },
        {
            "q": "Qu'est-ce que le chômage structurel ?",
            "k": "Le chômage structurel résulte d'inadéquations entre compétences des travailleurs et besoins du marché, souvent causées par des mutations technologiques. Contrairement au chômage conjoncturel, il persiste même en période de croissance. Il nécessite des politiques actives : formation professionnelle, reconversion et réforme du marché du travail.",
        },
    ],
    "informatique": [
        {
            "q": "Qu'est-ce que l'intelligence artificielle ?",
            "k": "L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine : raisonnement, apprentissage, perception et prise de décision. Elle se divise en IA faible (spécialisée) et IA générale (hypothétique). Les sous-domaines principaux sont le machine learning, le deep learning et le NLP.",
        },
        {
            "q": "Comment fonctionne le machine learning ?",
            "k": "Le machine learning est une branche de l'IA où les algorithmes apprennent automatiquement à partir de données. Les trois paradigmes : apprentissage supervisé (données étiquetées), non supervisé (détection de patterns) et par renforcement (essais/erreurs avec récompenses). Il alimente la recommandation, la détection de fraude et la vision par ordinateur.",
        },
        {
            "q": "Qu'est-ce que Python ?",
            "k": "Python est un langage de programmation interprété, orienté objet créé par Guido van Rossum en 1991. Sa syntaxe claire le rend accessible. Il est massivement utilisé en data science (NumPy, Pandas), IA/ML (TensorFlow, PyTorch), développement web (Django, Flask) et automatisation. Son écosystème PyPI compte plus de 400 000 packages.",
        },
        {
            "q": "Qu'est-ce que la cybersécurité ?",
            "k": "La cybersécurité protège les systèmes, réseaux et données contre les attaques numériques. Ses trois piliers : Confidentialité, Intégrité, Disponibilité (CIA). Les menaces principales : malwares, phishing, injections SQL, attaques DDoS et zero-days. Les protections : chiffrement (AES, RSA), pare-feux, authentification multi-facteurs et sensibilisation.",
        },
        {
            "q": "Qu'est-ce que le cloud computing ?",
            "k": "Le cloud fournit des ressources informatiques (serveurs, stockage, bases de données, IA) via internet à la demande. Les modèles : IaaS (infrastructure), PaaS (plateforme), SaaS (logiciel). Fournisseurs principaux : AWS, Azure, Google Cloud. Avantages : élasticité, paiement à l'usage, réduction des coûts CapEx.",
        },
        {
            "q": "Qu'est-ce qu'un algorithme et la complexité algorithmique ?",
            "k": "Un algorithme est une suite finie d'instructions pour résoudre un problème. Il possède des entrées, des sorties et se termine toujours. La complexité mesure ses ressources en notation Big-O. Exemples : tri rapide O(n log n), recherche binaire O(log n), Dijkstra pour les plus courts chemins. Les algorithmes sont le cœur de tout logiciel.",
        },
        {
            "q": "Qu'est-ce que le deep learning et les réseaux de neurones ?",
            "k": "Le deep learning utilise des réseaux de neurones à multiples couches cachées pour apprendre des représentations hiérarchiques. Les architectures principales : CNN (vision), RNN/LSTM (séquences) et Transformers (langage naturel). Il nécessite de grandes données et des GPU. Il alimente la reconnaissance faciale, les assistants vocaux et la traduction automatique.",
        },
        {
            "q": "Qu'est-ce que DevOps et CI/CD ?",
            "k": "DevOps unifie développement et opérations pour accélérer la livraison logicielle. Il repose sur l'intégration continue (CI), la livraison continue (CD), l'infrastructure as code et la surveillance. Outils : Docker, Kubernetes, Jenkins, GitLab CI/CD. L'objectif est de déployer plus vite, plus fréquemment et avec moins d'erreurs.",
        },
    ],
    "politique": [
        {
            "q": "Qu'est-ce que la démocratie ?",
            "k": "La démocratie est un système politique où le pouvoir appartient au peuple, exercé via des représentants élus. Ses fondements : suffrage universel, séparation des pouvoirs (législatif, exécutif, judiciaire), état de droit, presse libre et protection des droits fondamentaux. Elle s'oppose aux régimes autoritaires et totalitaires.",
        },
        {
            "q": "Comment fonctionne l'Union européenne ?",
            "k": "L'UE est une organisation supranationale de 27 États membres partageant un marché unique et une monnaie commune (euro). Ses institutions : Parlement européen, Conseil de l'UE, Commission européenne, Cour de justice et BCE. Fondée sur les Traités de Rome (1957), elle représente l'expérience d'intégration régionale la plus avancée.",
        },
        {
            "q": "Qu'est-ce que le populisme ?",
            "k": "Le populisme oppose 'le peuple pur' à 'l'élite corrompue'. Il peut être de gauche (critique des élites économiques : Chávez, Mélenchon) ou de droite (critique des élites culturelles et immigration : Trump, Orbán). Il prospère dans les contextes de crise de confiance institutionnelle et d'inégalités croissantes.",
        },
        {
            "q": "Qu'est-ce que la géopolitique contemporaine ?",
            "k": "La géopolitique étudie les relations entre puissance politique, territoire et ressources. Aujourd'hui : rivalité sino-américaine pour l'hégémonie, guerre en Ukraine, multilatéralisme affaibli, montée des puissances régionales (Inde, Brésil) et nouvelles conflictualités : cyberespace, désinformation et compétition technologique.",
        },
        {
            "q": "Qu'est-ce que la séparation des pouvoirs ?",
            "k": "Théorisée par Montesquieu dans L'Esprit des lois (1748), elle divise l'autorité étatique en trois branches : le législatif (vote les lois), l'exécutif (les applique) et le judiciaire (les interprète). Ce principe prévient la concentration du pouvoir. Dans les régimes présidentiels (USA), la séparation est stricte ; dans les régimes parlementaires, les deux premières sont liées.",
        },
    ],
    "sante": [
        {
            "q": "Comment fonctionne le système immunitaire ?",
            "k": "Le système immunitaire protège contre les pathogènes. L'immunité innée (rapide, non spécifique) mobilise macrophages et cellules NK. L'immunité adaptative (plus lente mais précise) produit des anticorps via les lymphocytes B et des cellules tueuses via les lymphocytes T. La mémoire immunitaire explique l'efficacité des vaccins.",
        },
        {
            "q": "Qu'est-ce que le diabète ?",
            "k": "Le diabète est une maladie chronique caractérisée par une hyperglycémie. Le type 1 (auto-immun) nécessite des injections d'insuline. Le type 2 (résistance à l'insuline) représente 90% des cas et se traite par alimentation équilibrée, activité physique et médicaments. Non contrôlé, il cause des complications cardiovasculaires, rénales et oculaires.",
        },
        {
            "q": "Qu'est-ce que la santé mentale ?",
            "k": "La santé mentale englobe le bien-être émotionnel, psychologique et social. Les troubles courants : dépression, anxiété, trouble bipolaire. Les facteurs protecteurs : liens sociaux, activité physique, sommeil (7-9h), pleine conscience. La thérapie cognitivo-comportementale (TCC) est la plus validée scientifiquement.",
        },
        {
            "q": "Comment fonctionnent les vaccins ?",
            "k": "Les vaccins stimulent le système immunitaire à reconnaître un pathogène sans causer la maladie. Ils présentent un antigène (virus atténué, protéine ou ARNm) qui déclenche la production d'anticorps et une mémoire immunitaire. Les vaccins à ARNm (Pfizer, Moderna) représentent une révolution technologique permettant un développement ultra-rapide.",
        },
        {
            "q": "Qu'est-ce qu'une alimentation saine ?",
            "k": "Une alimentation saine repose sur la diversité et l'équilibre : glucides complexes (45-55%), lipides sains (30-35%), protéines (15-20%). L'OMS recommande 5 portions de fruits et légumes par jour, des grains entiers et la limitation des sucres ajoutés, graisses saturées et sel. Le régime méditerranéen est régulièrement classé parmi les plus sains.",
        },
        {
            "q": "Comment fonctionne l'IRM ?",
            "k": "L'IRM (Imagerie par Résonance Magnétique) utilise un champ magnétique intense et des ondes radio pour imager les tissus mous sans rayonnements ionisants. Les protons d'hydrogène s'alignent dans le champ et émettent un signal qui est reconstruit en image 3D. L'IRMf mesure l'activité cérébrale via les variations du flux sanguin.",
        },
    ],
    "culture": [
        {
            "q": "Qui était Albert Einstein ?",
            "k": "Albert Einstein (1879-1955) était un physicien théoricien germano-américain. En 1905, il publia la relativité restreinte (E=mc²) et l'effet photoélectrique (Nobel 1921). En 1915, la relativité générale décrit la gravitation comme courbure de l'espace-temps. Il prédit les ondes gravitationnelles (confirmées 2015) et les trous noirs.",
        },
        {
            "q": "Qu'est-ce que la philosophie stoïcienne ?",
            "k": "Le stoïcisme, fondé par Zénon vers 300 av. J.-C., enseigne la dichotomie du contrôle : distinguer ce qui dépend de nous (jugements, actions) de ce qui n'en dépend pas (richesse, réputation). Le bonheur réside dans la vertu. Marc Aurèle, Épictète et Sénèque sont ses figures emblématiques. Il connaît un grand renouveau contemporain.",
        },
        {
            "q": "Qu'est-ce que la Révolution française ?",
            "k": "La Révolution française (1789-1799) abolit la monarchie absolue et l'Ancien Régime. Elle débute avec la prise de la Bastille (14 juillet 1789), la Déclaration des droits de l'homme, puis enchaîne la Terreur (Robespierre) et se termine par le coup d'État de Bonaparte. Elle diffusa les idéaux de Liberté, Égalité, Fraternité dans le monde.",
        },
        {
            "q": "Qui était Napoléon Bonaparte ?",
            "k": "Napoléon Bonaparte (1769-1821) fut Empereur des Français (1804-1814). Stratège militaire de génie, il vainquit la quasi-totalité de l'Europe avant d'échouer en Russie (1812) et à Waterloo (1815). Son héritage civil est immense : Code civil, Conseil d'État, Légion d'honneur, baccalauréat et Banque de France.",
        },
        {
            "q": "Qu'est-ce que l'impressionnisme ?",
            "k": "L'impressionnisme est un mouvement pictural né en France dans les années 1860-70. Ses artistes (Monet, Renoir, Degas, Morisot) peignaient en plein air, capturant la lumière et le mouvement par des touches rapides et des couleurs vives. Il libéra l'art occidental du réalisme et ouvrit la voie à l'art moderne.",
        },
    ],
    "sciences": [
        {
            "q": "Qu'est-ce que la mécanique quantique ?",
            "k": "La mécanique quantique décrit le comportement de la matière à l'échelle atomique. Ses principes : dualité onde-corpuscule, principe d'incertitude d'Heisenberg (impossible de connaître simultanément position et impulsion), superposition (plusieurs états simultanés jusqu'à la mesure). Elle est à la base des transistors, lasers et IRM.",
        },
        {
            "q": "Comment fonctionne la photosynthèse ?",
            "k": "La photosynthèse convertit l'énergie solaire en énergie chimique. Dans les chloroplastes, la lumière est absorbée par la chlorophylle. La phase lumineuse décompose l'eau et libère O₂. Le cycle de Calvin fixe le CO₂ pour synthétiser du glucose. Équation : 6CO₂ + 6H₂O + lumière → C₆H₁₂O₆ + 6O₂. Elle est la base de toute vie.",
        },
        {
            "q": "Qu'est-ce que l'ADN et la génétique ?",
            "k": "L'ADN est la molécule support de l'information génétique, en double hélice. Il est composé de quatre bases : adénine (A), thymine (T), guanine (G) et cytosine (C). Les gènes sont des séquences codant des protéines. CRISPR-Cas9 permet l'édition précise du génome, ouvrant des perspectives en médecine (thérapies géniques).",
        },
        {
            "q": "Qu'est-ce que le changement climatique ?",
            "k": "Le changement climatique désigne l'altération à long terme des températures mondiales. Les activités humaines (combustion de fossiles, déforestation) ont augmenté les gaz à effet de serre, amplifiant l'effet de serre. La température a déjà augmenté de +1,2°C. Conséquences : montée des eaux, événements extrêmes, acidification des océans.",
        },
        {
            "q": "Comment se forment les étoiles ?",
            "k": "Les étoiles se forment dans des nébuleuses de gaz qui s'effondrent gravitationnellement. Quand la température centrale atteint ~15 millions de degrés, la fusion nucléaire s'amorce (hydrogène → hélium). L'étoile entre en séquence principale. Notre système solaire s'est formé ainsi il y a 4,6 milliards d'années.",
        },
        {
            "q": "Qu'est-ce qu'un trou noir ?",
            "k": "Un trou noir est une région de l'espace-temps où la gravité est si intense que rien (pas même la lumière) ne peut s'en échapper. Il se forme par l'effondrement gravitationnel d'une étoile massive. L'horizon des événements est la frontière de non-retour. La première image d'un trou noir a été capturée par l'Event Horizon Telescope en 2019.",
        },
    ],
    "technologie": [
        {
            "q": "Comment fonctionnent les énergies renouvelables ?",
            "k": "Les énergies renouvelables proviennent de sources naturellement reconstituées : solaire (cellules photovoltaïques), éolien (turbines), hydraulique, géothermique et biomasse. En 2023, elles représentaient ~30% de la production mondiale d'électricité. Le coût du solaire a chuté de 90% en 10 ans. Le principal défi reste le stockage (batteries, hydrogène vert).",
        },
        {
            "q": "Qu'est-ce que l'informatique quantique ?",
            "k": "L'informatique quantique exploite la superposition et l'intrication quantiques. Les qubits représentent 0 et 1 simultanément. L'algorithme de Shor factoriserait des nombres géants en minutes (menaçant le chiffrement RSA). IBM, Google et IonQ développent des processeurs quantiques. La décoherence des qubits reste le principal défi.",
        },
        {
            "q": "Qu'est-ce que la réalité virtuelle et augmentée ?",
            "k": "La réalité virtuelle (VR) immerge totalement l'utilisateur dans un monde numérique (Meta Quest, PlayStation VR). La réalité augmentée (AR) superpose des éléments numériques au monde réel (HoloLens, Pokémon GO). Applications : jeux, formation professionnelle, chirurgie guidée, e-commerce et architecture.",
        },
        {
            "q": "Qu'est-ce que l'Industrie 4.0 ?",
            "k": "L'Industrie 4.0 désigne la numérisation et l'automatisation avancée de la production. Ses piliers : IoT industriel, jumeaux numériques (répliques virtuelles), IA, robotique collaborative (cobots), impression 3D et big data analytique. Elle permet une production ultra-flexible et prédictive mais implique une reconversion des compétences.",
        },
    ],
    "societe": [
        {
            "q": "Qu'est-ce que le développement durable ?",
            "k": "Le développement durable (rapport Brundtland, 1987) répond aux besoins du présent sans compromettre ceux des générations futures. Trois piliers : économique (croissance inclusive), social (équité) et environnemental (préservation des écosystèmes). Les 17 ODD de l'ONU fixent des objectifs concrets à atteindre d'ici 2030.",
        },
        {
            "q": "Quel est l'impact des réseaux sociaux ?",
            "k": "Les réseaux sociaux (Facebook, Instagram, TikTok, X, LinkedIn) ont révolutionné la communication et le marketing. Impacts négatifs : désinformation (fake news amplifiées par les algorithmes), addiction, cyberharcèlement, bulles de filtre et effets sur la santé mentale. La régulation européenne (DSA) tente d'encadrer leurs pratiques.",
        },
        {
            "q": "Comment fonctionne l'éducation moderne ?",
            "k": "L'éducation moderne combine enseignement présentiel et outils numériques (e-learning, MOOC, classes inversées). Les pédagogies actives remplacent progressivement la transmission magistrale. Les défis : réduction des inégalités scolaires, adaptation aux élèves à besoins particuliers, enseignement de la pensée critique et préparation aux métiers de l'IA.",
        },
        {
            "q": "Qu'est-ce que la fracture numérique ?",
            "k": "La fracture numérique désigne les inégalités d'accès et d'usage des technologies numériques selon des critères géographiques, socioéconomiques, générationnels et de compétences. Ces inégalités amplifient les inégalités sociales existantes car la numérisation touche l'emploi, l'éducation, la santé et les services publics.",
        },
    ],
}

# ═══════════════════════════════════════════════════════════
# 3. DÉTECTION RAPIDE SMALL-TALK (sans BERT, < 2ms)
# ═══════════════════════════════════════════════════════════

_HELLO_RE = re.compile(
    r"^(bonjour|bonsoir|salut|coucou|hey|hello|hi|yo|wesh|cc|bjr|bsr|slt|ola|hola)\b",
    re.IGNORECASE
)
_BYE_RE = re.compile(
    r"^(au revoir|bye|ciao|à plus|a\+|tchao|bonne (journée|soirée|nuit)|à bientôt)\b",
    re.IGNORECASE
)
_EMOTION_RE = re.compile(
    r"(comment (tu vas|ça va|vous allez)|ça va\b|je (suis|me sens)|comment allez-vous|t[ue]s (bien|mal)|tu vas bien)",
    re.IGNORECASE
)
_BOT_RE = re.compile(
    r"(qui (es.tu|t.es|êtes.vous)|qu.est.ce (que tu|que vous)|tu es (quoi|un|une)|c.est quoi lumina|parle.moi de toi|tes capacités|comment tu fonctionnes)",
    re.IGNORECASE
)

def detect_small_talk(text: str) -> Optional[tuple]:
    """Détection regex ultrarapide des salutations, émotions, questions sur le bot."""
    t = text.strip().lower()
    if _HELLO_RE.search(t):
        return ("greeting", random.choice(CORPUS["greetings"]["responses"]["hello"]))
    if _BYE_RE.search(t):
        return ("goodbye", random.choice(CORPUS["greetings"]["responses"]["goodbye"]))
    if _BOT_RE.search(t):
        return ("about_bot", random.choice(CORPUS["about_bot"]["responses"]))
    if _EMOTION_RE.search(t):
        return ("emotion", random.choice(CORPUS["emotions"]["responses"]))
    return None


# ═══════════════════════════════════════════════════════════
# 4. PRÉTRAITEMENT NLP
# ═══════════════════════════════════════════════════════════

class TextPreprocessor:
    """Pipeline de nettoyage, tokenisation et normalisation du texte."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            fr = set(stopwords.words("french"))
            en = set(stopwords.words("english"))
        except Exception:
            fr, en = set(), set()
        # Conserver les mots interrogatifs importants
        preserved = {"comment","pourquoi","quand","où","qui","quoi","quel","quelle","est","pas","ne"}
        self.stop_words = (fr | en) - preserved

    def clean(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^\w\s\u00C0-\u017E?!']", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return unicodedata.normalize("NFC", text)

    def tokenize_and_filter(self, text: str) -> list:
        cleaned = self.clean(text)
        try:
            tokens = word_tokenize(cleaned, language="french")
        except Exception:
            tokens = cleaned.split()
        return [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t.isalpha() and len(t) > 2 and t not in self.stop_words
        ]

    def preprocess(self, text: str) -> str:
        tokens = self.tokenize_and_filter(text)
        return " ".join(tokens) if tokens else self.clean(text)


# ═══════════════════════════════════════════════════════════
# 5. BERT + LRU CACHE POUR LES EMBEDDINGS
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_bert_model() -> SentenceTransformer:
    """Charge le modèle BERT une seule fois (cache Streamlit persistant)."""
    return SentenceTransformer(BERT_MODEL_NAME)


@st.cache_data(show_spinner=False)
def build_corpus_index(_model: SentenceTransformer) -> tuple:
    """Construit l'index BERT du corpus (résultat mis en cache)."""
    flat = []
    for domain, entries in CORPUS.items():
        if domain in ("greetings", "emotions", "about_bot"):
            continue
        if isinstance(entries, list):
            for entry in entries:
                flat.append({
                    "domain": domain,
                    "question": entry["q"],
                    "knowledge": entry["k"],
                    "index_text": f"{entry['q']} {entry['k'][:200]}",
                })
    texts = [e["index_text"] for e in flat]
    embeddings = _model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=32,
    )
    return flat, embeddings


@lru_cache(maxsize=MAX_CACHE_SIZE)
def cached_encode(text: str, model_name: str) -> tuple:
    """Encode un texte avec BERT (LRU cache pour les requêtes fréquentes)."""
    model = st.session_state.get("bert_model")
    if model is None:
        return tuple()
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return tuple(emb[0].tolist())


# ═══════════════════════════════════════════════════════════
# 6. MOTEUR DE RECHERCHE SÉMANTIQUE
# ═══════════════════════════════════════════════════════════

class SemanticSearchEngine:
    """Recherche sémantique par similarité cosinus des embeddings BERT."""

    def __init__(self, model, corpus_flat, corpus_embeddings):
        self.model = model
        self.corpus_flat = corpus_flat
        self.corpus_embeddings = corpus_embeddings
        self.preprocessor = TextPreprocessor()

    def encode_query(self, query: str) -> np.ndarray:
        processed = self.preprocessor.preprocess(query)
        cached = cached_encode(processed, BERT_MODEL_NAME)
        if cached:
            return np.array(cached, dtype=np.float32).reshape(1, -1)
        return self.model.encode([processed], convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = TOP_K) -> list:
        query_emb = self.encode_query(query)
        sims = cosine_similarity(query_emb, self.corpus_embeddings)[0]
        query_tokens = set(self.preprocessor.tokenize_and_filter(query))
        boosted = []
        for i, sim in enumerate(sims):
            entry_tokens = set(self.preprocessor.tokenize_and_filter(self.corpus_flat[i]["question"]))
            overlap = len(query_tokens & entry_tokens)
            bonus = min(overlap * 0.03, 0.12)
            boosted.append((i, float(sim) + bonus, float(sim)))
        boosted.sort(key=lambda x: -x[1])
        return [
            {**self.corpus_flat[i], "confidence": bs, "raw_similarity": rs}
            for i, bs, rs in boosted[:top_k]
        ]


# ═══════════════════════════════════════════════════════════
# 7. GÉNÉRATION DYNAMIQUE DE RÉPONSES
# ═══════════════════════════════════════════════════════════

DOMAIN_INTROS = {
    "economie":     ["Sur le plan économique :", "En économie :", "D'un point de vue économique :"],
    "informatique": ["Techniquement parlant :", "En informatique :", "Sur le plan numérique :"],
    "politique":    ["D'un point de vue politique :", "En sciences politiques :", "Politiquement :"],
    "sante":        ["Sur le plan médical :", "En matière de santé :", "Médicalement :"],
    "culture":      ["Culturellement :", "Sur le plan historique :", "D'un point de vue culturel :"],
    "sciences":     ["La science nous apprend que :", "Scientifiquement :", "D'après la science :"],
    "technologie":  ["Sur le plan technologique :", "En matière d'innovation :", "Technologiquement :"],
    "societe":      ["Dans notre société :", "D'un point de vue sociétal :", "Sur le plan social :"],
}

CLOSINGS = [
    " 💡 N'hésitez pas à poser des questions complémentaires !",
    " ✨ Souhaitez-vous approfondir un aspect particulier ?",
    " 🔍 Je peux développer ce point si vous le souhaitez.",
    "",
    " Avez-vous des questions supplémentaires ?",
]

def generate_dynamic_response(entry: dict, confidence: float) -> str:
    """Génère une réponse dynamique et contextualisée selon le domaine."""
    domain = entry.get("domain", "culture")
    intros = DOMAIN_INTROS.get(domain, ["Voici la réponse :", "En réponse à votre question :"])
    intro = random.choice(intros)
    knowledge = entry["knowledge"]
    closing = random.choice(CLOSINGS)
    if confidence >= 0.75:
        return f"{intro}\n\n{knowledge}{closing}"
    elif confidence >= 0.60:
        return f"{knowledge}{closing}"
    else:
        return f"Je pense pouvoir vous aider :\n\n{knowledge}"

def get_uncertainty_response() -> str:
    """Réponse standardisée pour les cas de faible confiance (< seuil)."""
    return random.choice([
        "Je ne suis pas sûr de comprendre votre question. 🤔 Pourriez-vous la reformuler ?",
        "Je ne suis pas sûr de comprendre votre question. Essayez avec d'autres mots ! 💭",
        "Je ne suis pas sûr de comprendre votre question. Mon score de confiance est trop faible. 🙏",
    ])


# ═══════════════════════════════════════════════════════════
# 8. CLASSE PRINCIPALE DU CHATBOT
# ═══════════════════════════════════════════════════════════

class LuminaChat:
    """
    Chatbot intelligent combinant :
    1. Détection rapide small-talk (regex, < 2ms)
    2. Compréhension sémantique BERT (embeddings + cosine similarity)
    3. Génération dynamique de réponses contextuelles
    4. Cache LRU pour les embeddings fréquents
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = load_bert_model()
        st.session_state["bert_model"] = self.model
        self.corpus_flat, self.corpus_embeddings = build_corpus_index(self.model)
        self.engine = SemanticSearchEngine(self.model, self.corpus_flat, self.corpus_embeddings)

    def get_response(self, query: str) -> dict:
        """Pipeline principal : small-talk → BERT search → génération → fallback."""
        query = query.strip()

        # Étape 1 : Small-talk rapide (regex, sans BERT)
        st_result = detect_small_talk(query)
        if st_result:
            talk_type, talk_resp = st_result
            return {
                "answer": talk_resp, "confidence": 1.0,
                "domain": "small_talk", "found": True,
                "response_type": talk_type, "top_matches": [],
            }

        # Étape 2 : Recherche sémantique BERT
        matches = self.engine.search(query, top_k=TOP_K)
        if not matches:
            return {
                "answer": get_uncertainty_response(), "confidence": 0.0,
                "domain": "—", "found": False,
                "response_type": "uncertain", "top_matches": [],
            }

        best = matches[0]
        confidence = best["confidence"]

        # Étape 3 : Décision confiance
        if confidence >= self.confidence_threshold:
            answer = generate_dynamic_response(best, confidence)
            return {
                "answer": answer, "confidence": confidence,
                "domain": best.get("domain", "—"), "found": True,
                "response_type": "semantic", "top_matches": matches,
            }
        else:
            return {
                "answer": get_uncertainty_response(), "confidence": confidence,
                "domain": best.get("domain", "—"), "found": False,
                "response_type": "uncertain", "top_matches": matches,
            }


# ═══════════════════════════════════════════════════════════
# 9. CSS — Warm Ivory × Forest Green
# ═══════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,400&family=Inconsolata:wght@300;400;500&display=swap');

:root {
    --ivory:  #f7f3ed; --ivory2: #efe9df; --cream: #e8dfd2;
    --forest: #2d5016; --sage:   #5a7a3a; --sage2: #7a9a58;
    --sienna: #a0522d; --gold:   #c8952a;
    --text:   #2a2016; --muted:  #8a7a68;
    --border: rgba(45,80,22,0.18);
    --user-bg: linear-gradient(135deg,#2d5016,#4a7a28);
    --radius: 18px; --radius-sm: 10px;
    --shadow: 0 4px 24px rgba(45,80,22,0.10);
    --mono: 'Inconsolata',monospace; --serif: 'Fraunces',serif;
}
*,*::before,*::after{box-sizing:border-box;}
.stApp{background:var(--ivory)!important;font-family:var(--serif)!important;color:var(--text)!important;}
#MainMenu,footer,.stDeployButton,header{visibility:hidden!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--ivory2);}
::-webkit-scrollbar-thumb{background:var(--sage2);border-radius:5px;}

/* Header */
.lm-header{background:var(--forest);padding:20px 28px;border-radius:var(--radius) var(--radius) 0 0;
  display:flex;align-items:center;gap:14px;position:relative;overflow:hidden;}
.lm-header::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--gold),var(--sage2),var(--gold));
  background-size:200% 100%;animation:goldShimmer 3s linear infinite;}
@keyframes goldShimmer{0%{background-position:100%}100%{background-position:-100%}}
.lm-icon{width:44px;height:44px;background:linear-gradient(135deg,var(--gold),#e8b040);
  border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:22px;
  box-shadow:0 0 20px rgba(200,149,42,0.4);animation:breathe 4s ease-in-out infinite;}
@keyframes breathe{0%,100%{box-shadow:0 0 20px rgba(200,149,42,0.4);transform:scale(1);}
  50%{box-shadow:0 0 32px rgba(200,149,42,0.6);transform:scale(1.04);}}
.lm-title h1{font-family:var(--serif);font-size:1.4rem;font-weight:600;
  color:var(--ivory);margin:0;letter-spacing:-0.01em;font-style:italic;}
.lm-title p{font-family:var(--mono);font-size:0.65rem;color:rgba(247,243,237,0.5);
  margin:3px 0 0;letter-spacing:0.12em;text-transform:uppercase;}
.lm-pill{margin-left:auto;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.1em;
  text-transform:uppercase;background:rgba(200,149,42,0.15);color:var(--gold);
  border:1px solid rgba(200,149,42,0.35);padding:4px 10px;border-radius:20px;
  display:flex;align-items:center;gap:5px;}
.pdot{width:6px;height:6px;border-radius:50%;background:var(--gold);
  animation:pulseDot 1.6s ease-in-out infinite;}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.3;transform:scale(0.7)}}

/* Chat area */
.lm-chat{background:var(--ivory2);border-left:1px solid var(--border);border-right:1px solid var(--border);padding:6px 0;}

/* Messages */
.msg-row{display:flex;gap:10px;padding:12px 22px;
  animation:msgSlide 0.26s cubic-bezier(0.34,1.56,0.64,1);}
@keyframes msgSlide{from{opacity:0;transform:translateY(12px) scale(0.97);}
  to{opacity:1;transform:translateY(0) scale(1);}}
.msg-row.ur{flex-direction:row-reverse;}

.ava{width:32px;height:32px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:14px;}
.ava.u{background:var(--forest);color:var(--ivory);font-family:var(--mono);
  font-size:10px;font-weight:500;box-shadow:0 2px 10px rgba(45,80,22,0.3);}
.ava.b{background:linear-gradient(135deg,var(--gold),#e8b040);
  box-shadow:0 2px 10px rgba(200,149,42,0.35);}

.bubble{max-width:74%;padding:12px 16px;font-size:0.9rem;line-height:1.72;
  word-break:break-word;position:relative;}
.bubble.ub{background:var(--forest);color:rgba(247,243,237,0.92);
  border-radius:var(--radius) var(--radius-sm) var(--radius) var(--radius);
  box-shadow:0 4px 16px rgba(45,80,22,0.25);font-style:italic;}
.bubble.bb{background:#fff;color:var(--text);
  border-radius:var(--radius-sm) var(--radius) var(--radius) var(--radius);
  border:1px solid var(--cream);box-shadow:0 2px 12px rgba(45,80,22,0.08);}

.msg-meta{font-family:var(--mono);font-size:0.62rem;color:var(--muted);
  margin-top:4px;padding:0 4px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.ur .msg-meta{justify-content:flex-end;}

.tag{padding:1px 6px;border-radius:4px;font-family:var(--mono);font-size:0.58rem;
  text-transform:uppercase;letter-spacing:0.07em;display:inline-flex;align-items:center;gap:3px;}
.td{background:rgba(90,122,58,0.12);color:var(--sage);border:1px solid rgba(90,122,58,0.25);}
.th{background:rgba(90,122,58,0.12);color:var(--sage);border:1px solid rgba(90,122,58,0.25);}
.tm{background:rgba(200,149,42,0.12);color:var(--gold);border:1px solid rgba(200,149,42,0.25);}
.tl{background:rgba(160,82,45,0.12);color:var(--sienna);border:1px solid rgba(160,82,45,0.25);}
.tt{background:rgba(45,80,22,0.08);color:var(--forest);border:1px solid rgba(45,80,22,0.2);}

.cbar{height:2px;background:var(--cream);border-radius:2px;margin-top:8px;}
.cbarf{height:100%;border-radius:2px;transition:width 0.9s ease;}

/* Match cards */
.mc{background:var(--ivory);border:1px solid var(--cream);
  border-radius:var(--radius-sm);padding:9px 13px;margin-bottom:6px;font-size:0.78rem;}
.mc .mq{color:var(--text);font-weight:600;margin-bottom:4px;}
.mc .ms{font-family:var(--mono);color:var(--muted);font-size:0.65rem;}
.mcbar{height:2px;background:var(--cream);border-radius:2px;margin:5px 0;}

/* Welcome */
.welcome{text-align:center;padding:56px 24px 44px;
  display:flex;flex-direction:column;align-items:center;gap:10px;}
.welcome .wi{font-size:2.8rem;}
.welcome h2{font-family:var(--serif);font-size:1.3rem;font-weight:600;
  color:var(--forest);margin:0;font-style:italic;}
.welcome p{font-family:var(--mono);font-size:0.73rem;color:var(--muted);
  max-width:360px;line-height:1.85;margin:0;}
.topics{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-top:8px;}
.tc{font-family:var(--mono);font-size:0.62rem;background:white;color:var(--forest);
  border:1px solid var(--border);padding:4px 11px;border-radius:20px;
  text-transform:uppercase;letter-spacing:0.06em;}

/* Footer */
.lm-footer{background:var(--ivory2);border:1px solid var(--border);
  border-top:none;border-radius:0 0 var(--radius) var(--radius);padding:14px 22px;}

/* Input */
.stTextInput>div>div{background:white!important;border:1.5px solid var(--border)!important;
  border-radius:var(--radius-sm)!important;color:var(--text)!important;
  font-family:var(--serif)!important;transition:border-color 0.2s,box-shadow 0.2s;
  box-shadow:0 2px 12px rgba(45,80,22,0.08);}
.stTextInput>div>div:focus-within{border-color:var(--sage)!important;
  box-shadow:0 0 0 3px rgba(90,122,58,0.15)!important;}
.stTextInput input{color:var(--text)!important;font-family:var(--serif)!important;
  font-style:italic!important;font-size:0.92rem!important;}
.stTextInput input::placeholder{color:var(--muted)!important;font-style:italic!important;}

/* Buttons */
.stButton>button{background:white!important;border:1.5px solid var(--border)!important;
  color:var(--muted)!important;border-radius:var(--radius-sm)!important;
  font-family:var(--mono)!important;font-size:0.72rem!important;
  letter-spacing:0.04em;transition:all 0.2s;}
.stButton>button:hover{border-color:var(--sage)!important;color:var(--forest)!important;
  box-shadow:0 2px 10px rgba(45,80,22,0.15)!important;
  background:rgba(90,122,58,0.05)!important;}
.stButton>button[kind="primary"]{background:var(--forest)!important;border:none!important;
  color:var(--ivory)!important;font-weight:500!important;
  box-shadow:0 4px 14px rgba(45,80,22,0.35)!important;}
.stButton>button[kind="primary"]:hover{background:var(--sage)!important;color:white!important;}

/* Sidebar */
[data-testid="stSidebar"]{background:white!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .stMarkdown *{color:var(--text)!important;}

/* Expander */
.streamlit-expanderHeader{background:var(--ivory)!important;border:1px solid var(--cream)!important;
  border-radius:var(--radius-sm)!important;color:var(--muted)!important;
  font-family:var(--mono)!important;font-size:0.68rem!important;
  letter-spacing:0.06em;text-transform:uppercase;}
.streamlit-expanderContent{background:var(--ivory)!important;border:1px solid var(--cream)!important;
  border-top:none!important;border-radius:0 0 var(--radius-sm) var(--radius-sm)!important;}
hr{border-color:var(--cream)!important;}
</style>
"""

# ═══════════════════════════════════════════════════════════
# 10. HELPERS D'INTERFACE
# ═══════════════════════════════════════════════════════════

DOMAIN_LABELS = {
    "economie":"Économie","informatique":"Informatique","politique":"Politique",
    "sante":"Santé","culture":"Culture","sciences":"Sciences",
    "technologie":"Technologie","societe":"Société","small_talk":"Conversation",
}

def conf_tag_html(score: float, r_type: str) -> str:
    if r_type in ("greeting","goodbye","about_bot","emotion"):
        return '<span class="tag tt">💬 small-talk</span>'
    if score >= 0.72: return f'<span class="tag th">✓ {score:.0%}</span>'
    if score >= 0.50: return f'<span class="tag tm">~ {score:.0%}</span>'
    return f'<span class="tag tl">✗ {score:.0%}</span>'

def bar_color(s: float) -> str:
    if s >= 0.72: return "#5a7a3a"
    if s >= 0.50: return "#c8952a"
    return "#a0522d"

def render_user(text, ts):
    st.markdown(f"""
    <div class="msg-row ur">
      <div class="ava u">Vous</div>
      <div>
        <div class="bubble ub">{text}</div>
        <div class="msg-meta">{ts}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_bot(text, ts, conf, domain, r_type, found):
    dom_tag = f'<span class="tag td">◉ {DOMAIN_LABELS.get(domain, domain)}</span>' if found else ""
    c_tag   = conf_tag_html(conf, r_type)
    bw      = int(min(conf, 1.0) * 100)
    bc      = bar_color(conf)
    body    = text.replace("\n\n","<br><br>").replace("\n","<br>")
    st.markdown(f"""
    <div class="msg-row">
      <div class="ava b">✦</div>
      <div style="flex:1;min-width:0">
        <div class="bubble bb">
          {body}
          <div class="cbar"><div class="cbarf" style="width:{bw}%;background:{bc}"></div></div>
        </div>
        <div class="msg-meta">{dom_tag} {c_tag} &nbsp;·&nbsp; {ts}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_welcome():
    topics = ["Économie","Informatique","Politique","Santé","Sciences","Technologie","Culture","Société"]
    chips = "".join(f'<span class="tc">{t}</span>' for t in topics)
    st.markdown(f"""
    <div class="welcome">
      <div class="wi">✦</div>
      <h2>Bonjour, je suis Lumina</h2>
      <p>Posez-moi n'importe quelle question — je comprends votre intention
         grâce à BERT et génère des réponses dynamiques et contextuelles.</p>
      <div class="topics">{chips}</div>
    </div>""", unsafe_allow_html=True)

def render_matches(matches):
    for m in matches[:3]:
        w = int(m["confidence"] * 100)
        c = bar_color(m["confidence"])
        dom = DOMAIN_LABELS.get(m.get("domain",""), "—")
        st.markdown(f"""
        <div class="mc">
          <div class="mq">❓ {m['question']}</div>
          <div class="mcbar"><div style="width:{w}%;height:100%;background:{c};border-radius:2px"></div></div>
          <div class="ms">{m['confidence']:.4f} cosine · {dom}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 11. APPLICATION STREAMLIT PRINCIPALE
# ═══════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Lumina Chat", page_icon="✦",
                       layout="centered", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────
    if "messages"        not in st.session_state: st.session_state.messages = []
    if "stats"           not in st.session_state: st.session_state.stats = {"n":0,"found":0,"conf":[]}
    if "pending_query"   not in st.session_state: st.session_state.pending_query = ""
    if "input_counter"   not in st.session_state: st.session_state.input_counter = 0
    if "chatbot"  not in st.session_state:
        with st.spinner("✦  Chargement de Lumina…"):
            st.session_state.chatbot = LuminaChat(CONFIDENCE_THRESHOLD)

    bot: LuminaChat = st.session_state.chatbot

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ✦ &nbsp;Lumina · Paramètres")
        st.markdown("---")
        threshold = st.slider("Seuil de confiance BERT", 0.10, 0.90,
                               float(bot.confidence_threshold), step=0.05)
        bot.confidence_threshold = threshold
        show_sources = st.toggle("Afficher les sources BERT", True)
        st.markdown("---")
        st.markdown("### 📊 Session")
        s = st.session_state.stats
        c1, c2 = st.columns(2)
        c1.metric("Questions", s["n"])
        c2.metric("Résolues",  s["found"])
        if s["conf"]:
            st.metric("Confiance moy.", f"{np.mean(s['conf']):.0%}")
        st.markdown("---")
        st.markdown("### 📚 Domaines du corpus")
        for d, entries in CORPUS.items():
            if isinstance(entries, list):
                lbl = DOMAIN_LABELS.get(d, d.capitalize())
                st.markdown(f"<small style='color:#8a7a68'>▸ **{lbl}** &mdash; {len(entries)} entrées</small>",
                            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"<small style='color:#8a7a68;font-family:Inconsolata,monospace'>"
                    f"BERT : {BERT_MODEL_NAME}<br>Seuil : {threshold:.0%}<br>"
                    f"LRU cache : {MAX_CACHE_SIZE} requêtes</small>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🗑 Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.stats = {"n":0,"found":0,"conf":[]}
            st.rerun()

    # ── Header ─────────────────────────────────────────────
    st.markdown("""
    <div class="lm-header">
      <div class="lm-icon">✦</div>
      <div class="lm-title">
        <h1>Lumina</h1>
        <p>Assistant IA · BERT · Génération Dynamique</p>
      </div>
      <div class="lm-pill"><span class="pdot"></span> En ligne</div>
    </div>
    <div class="lm-chat">""", unsafe_allow_html=True)

    # ── Messages ───────────────────────────────────────────
    if not st.session_state.messages:
        render_welcome()
    else:
        for msg in st.session_state.messages:
            ts = msg.get("time", "")
            if msg["role"] == "user":
                render_user(msg["content"], ts)
            else:
                render_bot(msg["content"], ts, msg.get("confidence",0),
                           msg.get("domain","—"), msg.get("response_type","semantic"),
                           msg.get("found",False))
                if show_sources and msg.get("top_matches") and msg.get("response_type")=="semantic":
                    with st.expander("🔍 Correspondances BERT · similarité cosinus"):
                        render_matches(msg["top_matches"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Zone de saisie ─────────────────────────────────────
    # On utilise une clé dynamique (input_counter) pour vider le champ
    # après envoi sans déclencher de boucle infinie.
    st.markdown('<div class="lm-footer">', unsafe_allow_html=True)
    col_in, col_btn = st.columns([5, 1])
    input_key = f"inp_{st.session_state.input_counter}"
    with col_in:
        user_input = st.text_input("", key=input_key,
            placeholder="Posez votre question ou dites bonjour…",
            label_visibility="collapsed")
    with col_btn:
        send = st.button("Envoyer", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Suggestions rapides ────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sugs = ["Bonjour !","Comment fonctionne l'IA ?","Qu'est-ce que le PIB ?",
            "Explique la photosynthèse","Qui était Einstein ?","Qu'est-ce que la démocratie ?"]
    cols = st.columns(3)
    for i, sug in enumerate(sugs):
        if cols[i % 3].button(sug, key=f"s{i}_{st.session_state.input_counter}",
                               use_container_width=True):
            st.session_state.pending_query = sug

    # ── Collecte de la requête (texte tapé OU suggestion) ──
    query_to_process = ""
    if send and user_input.strip():
        query_to_process = user_input.strip()
    elif st.session_state.pending_query:
        query_to_process = st.session_state.pending_query
        st.session_state.pending_query = ""   # consommé immédiatement

    # ── Traitement (une seule fois grâce au flag pending_query) ───────────
    if query_to_process:
        ts = time.strftime("%H:%M")
        st.session_state.messages.append({"role":"user","content":query_to_process,"time":ts})

        is_fast = detect_small_talk(query_to_process) is not None
        with st.spinner("✦ Un instant…" if not is_fast else "✦ Traitement…"):
            result = bot.get_response(query_to_process)

        s = st.session_state.stats
        s["n"] += 1
        s["conf"].append(result["confidence"])
        if result["found"]: s["found"] += 1

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
            "domain": result["domain"],
            "response_type": result["response_type"],
            "found": result["found"],
            "top_matches": result.get("top_matches", []),
            "time": ts,
        })
        # Incrémenter le compteur recrée un champ vide (nouvelle key) → pas de loop
        st.session_state.input_counter += 1
        st.rerun()

# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
