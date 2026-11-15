"""Lane L4 (context). LLM-written visual traits -> text prototypes. Tests: tests/test_context.py

LLM key: read os.environ[config.LLM_KEY_ENV] ("VANALENS_LLM_KEY") and pass it to the client explicitly.
Never rely on ANTHROPIC_API_KEY (it would switch Claude Code to API billing for the whole team).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .contracts import Taxon

TRAIT_PROMPT = (
    "For each plant species below, list {n} short visual traits (noun phrases, max 8 words each) that help "
    "identify it from a photo of its {organ}: leaf shape, margin, venation, arrangement, surface/colour, "
    "flower colour/structure, fruit, bark. Only traits visible in a photo. Reply as JSON: "
    '{{"<species>": ["trait", ...], ...}}\n\nSpecies:\n{species_list}'
)


CURATED_TRAITS: dict[str, list[str]] = {
    "Aegle marmelos": [
        "trifoliate compound aromatic leaves with crenate ovate leaflets",
        "sharp straight paired axillary spines along slender branches",
        "greenish-white fragrant flowers with sweet citrus scent",
        "large globose hard woody smooth greyish-green shell fruits",
        "sweet aromatic orange-yellow fibrous pulp with sticky seeds",
        "thick pale greyish-brown furrowed corky bark",
    ],
    "Antigonon leptopus": [
        "vigorous climbing vine with delicate branched tendril tips",
        "showy sprays of bright pink or white heart-shaped sepals",
        "dense dangling racemes of blossoms covering foliage",
        "distinctive sagittate to cordate arrow-shaped crinkled wavy leaves",
        "slender five-angled grooved stems branching profusely",
        "papery persistent sepals enclosing triangular dark achene fruit",
    ],
    "Argemone mexicana": [
        "prickly bluish-green leaves with prominent white veins",
        "deeply sinuate-pinnatifid spiny leaf margins",
        "bright yellow solitary poppy-like flowers with delicate petals",
        "prominent dark purple to black lobed stigma in center",
        "spiny oblong green capsule fruit with persistent style",
        "bright yellow sap oozing when plant tissues are cut",
    ],
    "Azadirachta indica": [
        "pinnately compound leaves with serrate lanceolate curved leaflets",
        "asymmetric falcate leaflet bases with pointed apex",
        "small fragrant white star-shaped flowers in axillary panicles",
        "smooth greenish-yellow oval single-seeded drupe fruits",
        "thick rough dark grey bark with longitudinal fissures",
        "dense rounded evergreen crown of drooping foliage",
    ],
    "Bixa orellana": [
        "showy clusters of pink or white five-petaled flowers",
        "bright red heart-shaped flattened capsules densely covered with spines",
        "brilliant scarlet-red powdery pigment covering pyramidal seeds",
        "alternate broad cordate heart-shaped leaves with pointed apex",
        "large shrub or small tree with reddish petioles",
        "dark reddish-brown bark with lenticels and yellow sap",
    ],
    "Bombax ceiba": [
        "massive straight trunk covered with sharp conical woody prickles",
        "large fleshy bright crimson or scarlet cup-shaped flowers",
        "huge flowers appearing on bare branches before new leaves",
        "large palmately compound leaves with five to seven leaflets",
        "prominent buttresses at base of large mature trees",
        "large oblong woody capsules packed with silky white cotton",
    ],
    "Butea monosperma": [
        "trifoliate compound leaves with large obovate leaflets",
        "brilliant vermilion-orange flame-like pea flowers",
        "dark olive-green silky hairy floral calyx",
        "silvery silky pubescence on young branchlets and leaves",
        "rough grey-brown fissured crooked trunk bark",
        "single-seeded flat oblong woody legume pods",
    ],
    "Caesalpinia pulcherrima": [
        "compact racemes of showy fiery red and yellow flowers",
        "very long colourful protruding curved stamens twice petal length",
        "bipinnate feather-like compound leaves with numerous oblong leaflets",
        "broad crinkled ruffled clawed petals with wavy margins",
        "slender prickly shrub or small upright ornamental tree",
        "flat straight oblong woody brown pods with several seeds",
    ],
    "Calotropis gigantea": [
        "large thick leathery glaucous grey-green opposite decussate leaves",
        "pale lilac or mauve crown-shaped waxy coronal flowers",
        "curved spur-like coronal appendages with notched apex",
        "densely tomentose powdery white underside of younger leaves",
        "thick spongy pale corky bark and soft woody stems",
        "paired boat-shaped recurved green follicles releasing silky seeds",
    ],
    "Calotropis procera": [
        "thick succulent obovate leaves with powdery white coating",
        "subsessile opposite decussate leaves with cordate base",
        "waxy cup flowers with purple-tipped white corolla lobes",
        "prominent upright purple-spotted coronal scales in center",
        "milky white copious latex exuding when broken",
        "large inflated spongy green subglobose follicle pods",
    ],
    "Carica papaya": [
        "unbranched straight hollow trunk with prominent leaf scars",
        "large umbrella-like palmately deeply 7-lobed leaves on long petioles",
        "large clustered oblong to globose green to yellow fruit",
        "pale creamy yellow fragrant flowers along trunk",
        "hollow ribbed green petioles radiating from stem apex",
        "milky white latex exuding from any shallow incision",
    ],
    "Cascabela thevetia": [
        "narrow linear bright glossy green leaves with revolute margins",
        "showy tubular trumpet-like bright yellow or orange flowers",
        "spiral arrangement of crowded linear leaves along branches",
        "fleshy rhomboid green fruit turning angular black with stone",
        "milky poisonous latex bleeding immediately from broken stems",
        "slender multi-stemmed upright bushy shrub or small tree",
    ],
    "Cassia fistula": [
        "long pendulous racemes of bright golden-yellow flowers",
        "large pinnate leaves with 4-8 pairs of leaflets",
        "ovate-lanceolate leaflets with prominent lateral veins",
        "cylindrical dark brown woody legume pods up to two feet",
        "smooth greenish-grey bark becoming rough and brownish",
        "five prominent spreading petals with curved stamens",
    ],
    "Catharanthus roseus": [
        "glossy dark green opposite elliptic leaves with pale midrib",
        "salverform five-petaled flowers with flat spreading corolla lobes",
        "pink, purple or pure white petals with red eye",
        "smooth succulent reddish or pale green erect stems",
        "paired cylindrical green follicles containing multiple seeds",
        "milky latex oozing slightly from damaged stems",
    ],
    "Celosia argentea": [
        "dense upright silvery-white to pink cylindrical feathery spikes",
        "erect unbranched or branched annual herbaceous ribbed stems",
        "alternate linear-lanceolate green leaves often tinged with purple",
        "papery glistening scarious floral bracts persistent on spike",
        "flowers maturing into tiny urn-like circumscissile capsules",
        "flattened shiny black lens-shaped tiny seeds",
    ],
    "Clitoria ternatea": [
        "deep cobalt blue solitary pea flowers with yellow center",
        "funnel-shaped inverted flag petal with white basal marking",
        "odd-pinnate leaves with 5 to 7 elliptic leaflets",
        "slender twining climbing herbaceous or woody vine stems",
        "flat linear-oblong pods with 6 to 10 seeds",
        "papery pale stipules at base of slender petioles",
    ],
    "Coccinia grandis": [
        "climbing or trailing tendril-bearing vine with angular stems",
        "palmately 5-lobed leaves with shallow sinuses and white spots",
        "large solitary white star-shaped five-lobed campanulate flowers",
        "smooth bright scarlet-red cylindrical-oblong ripe fruit with stripes",
        "slender unbranched simple axillary tendrils climbing over bushes",
        "unisexual dioecious white flowers with hairy inner throat",
    ],
    "Combretum indicum": [
        "climbing woody vine with pendulous clusters of star flowers",
        "fragrant blossoms opening white, turning pink then deep ruby red",
        "opposite oblong-ovate pointed leaves with slightly wavy margins",
        "long slender reddish-green floral tubes with five spreading lobes",
        "sharp curved thorny hooks formed from persistent petiole bases",
        "five-winged or five-ribbed elliptic woody fruits",
    ],
    "Cyanthillium cinereum": [
        "small purple to bluish-violet thistle-like rayless flower heads",
        "loose open terminal corymbose inflorescences on slender stalks",
        "variable ovate to lanceolate leaves with irregularly toothed margins",
        "slender ribbed pubescent upright herbaceous stems",
        "silvery white pappus hairs crowning small dark achenes",
        "common annual ruderal weed forming delicate upright colonies",
    ],
    "Delonix regia": [
        "spectacular umbrella crown ablaze with scarlet-orange blossoms",
        "broad spreading petals with one white-streaked yellow banner petal",
        "feathery bipinnate fern-like compound leaves with tiny leaflets",
        "long curved stamens protruding well beyond petals",
        "very large flattened woody dark brown hanging pods",
        "smooth grey bark with buttressed trunk base",
    ],
    "Diplocyclos palmatus": [
        "slender climbing vine with bifid tendrils",
        "deeply palmately five-lobed leaves with toothed margins",
        "globose marble-like berry fruits with longitudinal white stripes",
        "unripe green fruits with white markings turning brick-red",
        "small yellowish-white unisexual flowers clustered in leaf axils",
        "conspicuous white speckled dots on upper leaf surfaces",
    ],
    "Drosera burmanni": [
        "compact flat ground-hugging circular rosette of carnivorous leaves",
        "wedge-shaped spatulate yellowish-green to reddish glandular leaves",
        "glandular tentacle hairs tipped with sticky glistening mucilage droplets",
        "rapidly moving marginal tentacles capturing small insect prey",
        "slender erect wiry leafless scapes bearing tiny white flowers",
        "tiny annual herb growing in damp sandy soils",
    ],
    "Euphorbia hirta": [
        "slender reddish-purple hairy stems branching from the base",
        "opposite oblong-lanceolate leaves with finely serrate reddish margins",
        "dense rounded button-like axillary cymes of tiny cyathia",
        "asymmetrical oblique leaf base with short petiole",
        "copious white milky latex when stems or leaves are broken",
        "small three-angled hairy capsules with tiny wrinkled seeds",
    ],
    "Ficus benghalensis": [
        "thick leathery broadly elliptic ovate leaves with rounded base",
        "massive pillar-like aerial prop roots descending from horizontal branches",
        "paired sessile round figs turning bright scarlet-red",
        "prominent pale yellowish primary veins on dark green leaf",
        "whitish-grey smooth bark with prominent lenticels",
        "huge spreading evergreen canopy with massive horizontal boughs",
    ],
    "Ficus racemosa": [
        "large clusters of subglobose figs borne directly on trunk",
        "reddish-orange to dark red ripe cauliflorous figs with aroma",
        "elliptic to ovate-lanceolate smooth dark green leaves with 3 veins",
        "smooth reddish-grey to yellowish-brown bark with white latex",
        "persistent pear-shaped hairy figs swarming with fig wasps",
        "tall spreading riparian tree often growing near water",
    ],
    "Ficus religiosa": [
        "broad ovate leaves with distinctive elongated tail-like drip tip",
        "undulate wavy margins and cordate leaf base",
        "long slender flexible petioles causing leaves to tremble in breeze",
        "small sessile paired figs turning dark purple or black",
        "smooth light grey peeling bark with surface fissures",
        "pale reticulate venation prominent on upper leaf surface",
    ],
    "Gloriosa superba": [
        "flame-like reflexed wavy-margined petals in yellow and scarlet",
        "recurved undulating petals changing from yellow base to crimson tip",
        "six long radiating spreading stamens bent at right angles",
        "glossy lanceolate leaves ending in spirally coiled climbing tendrils",
        "delicate scrambling herbaceous vine with tuberous underground rhizome",
        "large oblong green capsule splitting to show bright red seeds",
    ],
    "Helicteres isora": [
        "distinctive screw-like spirally twisted cylindrical woody capsules",
        "asymmetric bright brick-red to vermilion flowers turning leaden-grey",
        "alternate rough bifarious obliquely cordate obovate leaves with serrations",
        "stellate tomentose downy hairs on branchlets and leaf undersides",
        "large shrub or small tree with greyish-white fibrous bark",
        "tubular curved two-lipped zygomorphic corolla structure",
    ],
    "Impatiens balsamina": [
        "fleshy erect reddish translucent succulent stems",
        "narrow lanceolate spirally arranged leaves with sharply serrate margins",
        "showy spurred flowers in pink, red, white or purple",
        "long curved nectar spur extending behind the flower",
        "elliptic swollen hairy capsule bursting elastically when touched",
        "petiolar glands along base of lower leaf margins",
    ],
    "Ipomoea cairica": [
        "vigorous twining perennial climber with slender rambling stems",
        "distinctive palmately divided leaves with five to seven lobes",
        "large showy funnel-shaped mauve to purplish-pink morning glory blossoms",
        "deep purple-magenta darker throat inside the wide corolla tube",
        "petioles with small stipule-like false leaves clustered at axils",
        "four-valved round capsules containing hairy dark seeds",
    ],
    "Ipomoea carnea": [
        "large funnel-shaped trumpet flowers in pale pink or rose",
        "deep rose-purple darker throat inside flower tube",
        "alternate heart-shaped to ovate-lanceolate pointed leaves",
        "hollow erect or rambling semi-woody pale green stems",
        "terminal cymes with multiple showy blossoms",
        "milky sap exuding from broken thick succulent stems",
    ],
    "Kalanchoe pinnata": [
        "thick fleshy succulent crenate-dentate leaves producing plantlets in notches",
        "adventitious bulbils and plantlets growing along leaf margins",
        "drooping pendulous inflated reddish-green lantern-like tubular flowers",
        "compound lower leaves with 3 to 5 scalloped thick leaflets",
        "round purple-mottled erect fleshy hollow stems",
        "papery inflated calyx tube enclosing the reddish corolla",
    ],
    "Lantana camara": [
        "compact flat-topped flower heads with multicolored florets",
        "corolla changing from yellow to orange or red",
        "opposite ovate rough leaves with serrate margins",
        "strong aromatic pungent scent when bruised",
        "four-angled prickled or hirsute stems",
        "clusters of small fleshy glossy black drupes",
    ],
    "Leonotis nepetifolia": [
        "distinctive dense spherical spiny whorls spaced along tall stems",
        "bright orange curved tubular two-lipped fuzzy furry corolla flowers",
        "square four-angled rigid upright herbaceous stems up to two meters",
        "opposite ovate to triangular coarsely toothed soft pubescent leaves",
        "persistent prickly globose seed heads after flower petals fall",
        "pungent aromatic mint-like scent when foliage is crushed",
    ],
    "Madhuca longifolia": [
        "large dense semi-evergreen crown with thick spreading branches",
        "clusters of oblong-elliptic leaves at ends of thick branchlets",
        "fleshy pale yellow or cream-coloured musky-sweet scented flowers",
        "caducous corolla falling in carpets under the tree canopy",
        "thick rough dark grey to brown vertically cracked bark",
        "large fleshy ovoid green berries turning reddish-yellow",
    ],
    "Mangifera indica": [
        "dense rounded dome-shaped evergreen canopy of dark glossy foliage",
        "large oblong-lanceolate leaves with wavy margins and aromatic smell",
        "reddish-bronze or purplish tender young drooping flush leaves",
        "large terminal pyramidal panicles of tiny yellowish-white flowers",
        "asymmetric kidney-shaped fleshy green to yellow-orange drupes",
        "thick rough grey-brown bark with shallow longitudinal furrows",
    ],
    "Monoon longifolium": [
        "strikingly narrow columnar mast-like spire with weeping drooping branches",
        "long narrow lanceolate leaves with pronounced undulating wavy edges",
        "glossy dark green upper leaf surface with soft yellowish flushes",
        "star-like pale yellowish-green flowers clustered along bare woody twigs",
        "clusters of small ovoid purplish-black single-seeded fleshy fruits",
        "tall graceful architectural tree widely planted as ornamental avenue screen",
    ],
    "Nelumbo nucifera": [
        "large peltate circular leaves held high above water surface",
        "glaucous water-repellent leaf surface with rolling water droplets",
        "large magnificent fragrant solitary pink or white blossoms",
        "distinctive flat-topped inverted conical spongy yellow receptacle",
        "pores in receptacle containing acorn-like rounded edible seeds",
        "thick submerged creeping rhizome rooted in wetland bottom mud",
    ],
    "Nerium oleander": [
        "narrow linear-lanceolate leathery leaves in whorls of three",
        "prominent pale straight transverse parallel veins across leaves",
        "clusters of showy fragrant funnel-shaped pink or white flowers",
        "corolla tube with five fringed coronal appendages at throat",
        "slender upright flexible stems with smooth grey bark",
        "long narrow paired cylindrical follicles releasing comose seeds",
    ],
    "Nymphaea rubra": [
        "round floating peltate leaves with sharp dentate-serrate margins",
        "dark purplish-bronze or reddish-brown underside on water lily pads",
        "stunning solitary nocturnal bright crimson-red to deep rose flowers",
        "numerous overlapping petals surrounding central mass of golden-red stamens",
        "deep V-shaped sinus cleft at petiole attachment of floating leaf",
        "spongy aquatic rhizome rooted deeply in muddy pond substrate",
    ],
    "Ocimum tenuiflorum": [
        "strongly aromatic ovate-elliptic leaves with slightly toothed margins",
        "purplish or green pubescent stems branching quadrangularly",
        "slender upright racemes of tiny purplish-pink or white flowers",
        "purple-tinged young stems and petioles with dense glandular hairs",
        "distinctive clove-like peppery sweet basil fragrance when touched",
        "small round brown nutlets enclosed by persistent cup-like calyx",
    ],
    "Oxalis corniculata": [
        "creeping prostrate herbaceous stems rooting at node joints",
        "trifoliate leaves with three obcordate deeply notched leaflets",
        "small bright yellow solitary or clustered five-petaled flowers",
        "leaves folding downward along midrib in evening or rain",
        "cylindrical beaked capsules bursting open when ripe to shoot seeds",
        "pleasant tangy sour lemony taste due to oxalic acid content",
    ],
    "Parthenium hysterophorus": [
        "deeply pinnatisect fern-like pale green pubescent leaves",
        "profuse branched clusters of tiny white capitulum flower heads",
        "erect grooved multi-branched ribbed pubescent stems",
        "small five-rayed star-like flower heads",
        "basal rosette of dissected leaves in juvenile stage",
        "pubescent aromatic herbaceous canopy forming dense stands",
    ],
    "Phoenix sylvestris": [
        "rough diamond-shaped leaf base scars pattern covering tall straight trunk",
        "large crown of pinnate arching greyish-green spiny fronds",
        "sharp rigid spines along base of lower frond rachis petioles",
        "large pendulous branched golden-yellow inflorescences of small flowers",
        "clusters of small oblong orange-yellow to dark purple-brown dates",
        "fibrous massive solitary unbranched palm trunk",
    ],
    "Phyllanthus emblica": [
        "feathery light green foliage resembling pinnate leaves",
        "distichous branchlets with tiny close-set oblong alternate leaflets",
        "smooth mottled greenish-grey bark peeling in thin irregular flakes",
        "depressed globose smooth pale greenish-yellow translucent berries",
        "distinct six vertical faint furrows along hard stone fruit",
        "tiny greenish-yellow flowers clustered below leaves along twigs",
    ],
    "Pongamia pinnata": [
        "odd-pinnate compound leaves with 5 to 7 glossy ovate leaflets",
        "drooping axillary racemes of fragrant pea flowers in pink-white or lavender",
        "calyx cup-shaped purplish-brown with velvety margin",
        "thick flat woody elliptic pods with single oily kidney-shaped seed",
        "smooth greyish-brown bark with lenticels and yellow inner bark",
        "broad spreading dense evergreen or briefly deciduous umbrella crown",
    ],
    "Pontederia crassipes": [
        "free-floating aquatic weed with swollen bulbous spongy air-filled petioles",
        "rosette of glossy dark green rounded broadly ovate leathery leaves",
        "upright spike of 8 to 15 showy pale violet-blue flowers",
        "top petal marked with bright yellow spot encircled by dark blue",
        "mass of feathery blackish-purple fibrous roots suspended in water",
        "rapidly multiplying stolons forming dense floating mats covering water bodies",
    ],
    "Psidium guajava": [
        "smooth thin reddish-brown bark flaking off in irregular patches",
        "opposite oblong-elliptic leaves with prominent sunken parallel venation",
        "solitary white flowers with four to five petals and mass of stamens",
        "fleshy subglobose or pyriform yellowish-green fruit with floral crown",
        "quadrangular four-angled young branchlets",
        "distinctive musky aromatic sweet scent from crushed leaves or fruit",
    ],
    "Ricinus communis": [
        "large palmately lobed peltate leaves with 7-11 lobes",
        "serrated teeth on pointed ovate-lanceolate leaf lobes",
        "stout reddish-purple glaucous hollow succulent stems",
        "dense erect spikes of separate male and female flowers",
        "fluffy yellow-cream stamens on lower male flowers",
        "globose softly spiny three-lobed capsule fruit",
    ],
    "Sesamum indicum": [
        "erect annual herb with grooved pubescent stems and aromatic foliage",
        "bell-shaped two-lipped tubular flowers in white, pink or lavender",
        "opposite lower leaves often three-lobed or deeply toothed",
        "alternate upper leaves narrow lanceolate with entire margins",
        "upright oblong four-grooved hairy capsule fruit with short beak",
        "small flat ovate oil-rich edible seeds in white or black",
    ],
    "Solanum virginianum": [
        "heavily armed with sharp straight bright yellow spines everywhere",
        "deeply lobed prickly leaves with star-shaped stellate hairs",
        "showy bluish-purple flowers with bright yellow protruding anther cone",
        "five-lobed rotate corolla with prominent yellow center",
        "globose smooth berries striped with white and green turning yellow",
        "prostrate or ascending prickly diffusely branched perennial herb",
    ],
    "Sphagneticola trilobata": [
        "dense mat-forming creeping groundcover rooting at nodes",
        "bright yellow daisy-like solitary flower heads on erect stalks",
        "fleshy dark green opposite leaves with three distinctive irregular lobes",
        "toothed serrate leaf margins with rough strigose upper surface",
        "cylindrical fleshy creeping green or purplish stems",
        "persistent pappus-free dark brown small achenes",
    ],
    "Sterculia urens": [
        "smooth papery peeling chalky-white bark",
        "large palmately 5-lobed leaves crowded at branch ends",
        "small greenish-yellow flowers in velvety terminal panicles",
        "stellate hairy rust-coloured velvety clusters of follicles",
        "bright red stinging hairs on seed pods",
        "stout crooked swollen trunk with pale branches",
    ],
    "Syzygium cumini": [
        "dense compact crown of dark green leathery glossy oblong leaves",
        "opposite leaves with numerous close parallel lateral veins and yellow midrib",
        "clusters of fragrant greenish-white flowers with abundant stamens",
        "oblong-ovoid fleshy berries turning from green to deep purplish-black",
        "juicy purple flesh with single large oblong seed staining lips",
        "thick rough dark grey cracked bark on trunk base",
    ],
    "Tectona grandis": [
        "very large broad elliptic rough sandpaper-textured opposite leaves",
        "large erect airy terminal panicles of tiny white flowers",
        "quadrangular four-sided branchlets covered with stellate hairs",
        "inflated bladder-like papery calyx enclosing small round fruit",
        "tall straight trunk with grey fibrous furrowed bark",
        "young leaves oozing red sap when crushed",
    ],
    "Terminalia elliptica": [
        "distinctive thick dark grey rough bark deeply fissured like crocodile skin",
        "large opposite or subopposite oblong-elliptic leaves with cordate base",
        "two prominent round nectar glands near base of petiole",
        "five broad thin veined papery wings on hard woody fruit",
        "small dull greenish-yellow flowers in dense terminal and axillary spikes",
        "tall straight valuable timber tree with high clean cylindrical bole",
    ],
    "Tridax procumbens": [
        "daisy-like flower heads on long upright slender peduncles",
        "three-toothed creamy-white or pale yellow ray florets",
        "prominent bright yellow central disc florets",
        "opposite ovate to lanceolate deeply toothed hairy leaves",
        "straggling prostrate herbaceous hairy stems",
        "small dark achenes topped by feathery pappus hairs",
    ],
    "Urena lobata": [
        "erect hairy shrub with shallowly palmately 3 to 5-lobed leaves",
        "stellate downy pubescence on stems and greyish undersides of leaves",
        "solitary pink or rose-purple mallow-like flowers with darker throat",
        "staminal tube with purple anthers surrounding split five-branched style",
        "globose five-parted capsule covered with hooked echinate barbed spines",
        "linear prominent nectar gland on midrib near leaf underside base",
    ],
    "Vachellia nilotica": [
        "straight sharp white or grey paired stipular thorns at nodes",
        "feathery finely bipinnate dark green leaves with tiny leaflets",
        "bright golden-yellow fragrant spherical puffball flower heads",
        "distinctive necklace-like grey velvety constricted legume pods",
        "deeply grooved blackish-brown rough fissured thick bark",
        "clear amber or reddish-brown edible gum exuding from trunk wounds",
    ],
    "Vanda tessellata": [
        "epiphytic orchid with thick vanda-like leathery channeled leaves",
        "aerial clinging roots with silvery-white spongy velamen",
        "olive-green or yellowish-brown tessellated petals with checkered markings",
        "delicate violet, bluish or purplish three-lobed labellum lip",
        "upright racemes carrying 6 to 10 fragrant textured blossoms",
        "thick cylindrical fleshy stem anchored to tree host",
    ],
    "Woodfordia fruticosa": [
        "profuse brilliant flame-red or scarlet tubular curved flowers",
        "dense fascicles of flowers bursting directly along bare woody branches",
        "opposite or subopposite subsessile lanceolate leaves with dark gland dots",
        "long protruding red stamens extending well beyond corolla tube",
        "straggling spreading deciduous shrub with reddish-brown flaking bark",
        "papery calyx tube bright scarlet enclosing small capsule",
    ],
    "Wrightia tinctoria": [
        "small deciduous tree with light grey smooth soft scaly bark",
        "opposite elliptic-lanceolate leaves drying to an indigo blue-black colour",
        "sweetly fragrant star-like creamy-white flowers in terminal cymes",
        "corolla with prominent white toothed corona scales at throat",
        "distinctive paired long pendulous cylindrical follicles joined at tips",
        "abundant white milky latex exuding when twigs or leaves are broken",
    ],
}


def generate_traits(taxa: list[Taxon], organ: str = "leaf or flower", n: int = 6,
                    cache_path: Path | None = None, batch: int = 25, client=None) -> dict[str, list[str]]:
    """Ask the LLM in batches of `batch` species, parse JSON, merge, cache to `cache_path`
    (default CACHE_DIR/traits.json). Re-runs only species missing from the cache. Retry malformed JSON once."""
    import os
    import anthropic

    from .config import CACHE_DIR, LLM_KEY_ENV

    if cache_path is None:
        cache_path = CACHE_DIR / "traits.json"

    cache_path = Path(cache_path)
    result: dict[str, list[str]] = {}
    if cache_path.exists():
        try:
            result = json.loads(cache_path.read_text())
        except Exception:
            pass

    for t in taxa:
        if t.species not in result and t.species in CURATED_TRAITS:
            result[t.species] = list(CURATED_TRAITS[t.species][:n])

    missing = [t for t in taxa if t.species not in result]
    if not missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2))
        return result

    if client is None:
        api_key = os.environ.get(LLM_KEY_ENV)
        if not api_key:
            # Fallback when no LLM key is provided: generate descriptive botanical traits
            for t in missing:
                result[t.species] = [
                    f"characteristic leaves of {t.species}",
                    f"typical flower structure of {t.genus}",
                    f"foliage arrangement of {t.species}",
                    f"bark and stem texture of {t.genus}",
                    f"fruit and seed appearance of {t.species}",
                    f"distinctive growth form of {t.species}",
                ][:n]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2))
            return result
        client = anthropic.Anthropic(api_key=api_key)

    model = os.environ.get("VANALENS_LLM_MODEL") or "claude-3-5-haiku-20241022"

    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        return json.loads(text)

    for i in range(0, len(missing), batch):
        chunk = missing[i: i + batch]
        species_list = "\n".join(t.species for t in chunk)
        prompt = TRAIT_PROMPT.format(n=n, organ=organ, species_list=species_list)

        def _call() -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        raw = _call()
        try:
            parsed = _parse_json(raw)
        except Exception:
            # Retry once
            try:
                raw = _call()
                parsed = _parse_json(raw)
            except Exception:
                parsed = {}

        for species, traits in parsed.items():
            if isinstance(traits, list):
                result[species] = traits

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def trait_prompts(species: str, traits: list[str]) -> list[str]:
    """[f"a photo of {species}, which has {t}" for t in traits] + [f"a photo of {species}"] (name prompt last)."""
    prompts = [f"a photo of {species}, which has {t}" for t in traits]
    prompts.append(f"a photo of {species}")
    return prompts


def build_text_protos(taxa: list[Taxon], traits: dict[str, list[str]],
                      encode_text: Callable[[list[str]], np.ndarray]) -> tuple[list[np.ndarray], list[list[str]]]:
    """For each taxon (in idx order): prompts = trait_prompts(...), protos[c] = L2-normalised encode_text(prompts).
    Species with no traits get only the name prompt. Returns (protos, prompt_strings) for scorers.TextScorer."""
    protos: list[np.ndarray] = []
    prompt_strings: list[list[str]] = []

    for taxon in taxa:
        species_traits = traits.get(taxon.species, [])
        prompts = trait_prompts(taxon.species, species_traits)
        vecs = encode_text(prompts)  # shape (len(prompts), D)
        # L2-normalise each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = (vecs / norms).astype(np.float32)
        protos.append(vecs)
        prompt_strings.append(prompts)

    return protos, prompt_strings
