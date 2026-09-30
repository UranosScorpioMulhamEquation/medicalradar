import datetime
import itertools
import os
from typing import Dict, List, Tuple, Any, Optional

import geonamescache
import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components
import swisseph as swe
from timezonefinder import TimezoneFinder

# ==========================================
# 0. إعداد مسار ملفات الـ Ephemeris
# ==========================================
EPHE_PATH = "./ephe"
if not os.path.exists(EPHE_PATH):
    os.makedirs(EPHE_PATH)

swe.set_ephe_path(EPHE_PATH)


# ==========================================
# 1. خدمات البيانات الجغرافية والزمنية
# ==========================================
@st.cache_data
def load_geo_data() -> Tuple[geonamescache.GeonamesCache, Dict[str, str], List[str], Dict[str, List[Dict[str, Any]]]]:
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()

    country_map = {c["name"]: code for code, c in countries.items()}
    sorted_countries = sorted(list(country_map.keys()))

    cities_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for c_code in country_map.values():
        c_cities = [city for city in cities.values() if city["countrycode"] == c_code]
        c_cities_sorted = sorted(c_cities, key=lambda x: x["population"], reverse=True)
        cities_by_country[c_code] = c_cities_sorted

    return gc, country_map, sorted_countries, cities_by_country


gc, country_map, country_list, cities_by_country = load_geo_data()


@st.cache_resource
def get_tz_finder() -> TimezoneFinder:
    return TimezoneFinder()


tf = get_tz_finder()


def get_utc_julian_day(date_obj: datetime.date, time_obj: datetime.time, tz_str: str) -> float:
    local_tz = pytz.timezone(tz_str)
    dt = datetime.datetime.combine(date_obj, time_obj)
    local_dt = local_tz.localize(dt)
    utc_dt = local_dt.astimezone(pytz.utc)
    return swe.julday(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0,
    )


def get_aspect(lon1: float, lon2: float, orb: float = 8.0) -> Optional[str]:
    diff = abs(lon1 - lon2)
    if diff > 180:
        diff = 360 - diff

    aspects = {
        "الاقتران": 0.0,
        "التسديس": 60.0,
        "التربيع": 90.0,
        "التثليث": 120.0,
        "المقابلة": 180.0,
    }

    for name, angle in aspects.items():
        if abs(diff - angle) <= orb:
            return name
    return None


# ==========================================
# 2. محرك التحليل الفلكي الطبي والنفسي المتطور
# ==========================================
class MedicalAstroAnalyzer:
    """
    محرك تحليل التنجيم الطبي والنفسي المستند إلى المراجع الكلاسيكية (خوان استاديا)
    وأحدث أبحاث التنجيم الطبي للكويكبات.
    """

    def __init__(self):
        # الدلالات الفسيولوجية والنفسية للكواكب
        self.planets = {
            "الشمس": {
                "ext": "الظهر (الجزء العلوي)، العيون (مع القمر)",
                "int": "القلب، العمود الفقري العلوي، الدورة الدموية الشريانية",
                "diseases": "أمراض القلب، ضعف الحيوية، الإجهاد الحراري",
                "psych": "تقدير الذات، الأنا، مستويات الطاقة النفسية، الإرهاق العاطفي (Burnout)"
            },
            "القمر": {
                "ext": "الصدر، الثدي، العيون",
                "int": "المعدة، السوائل الجسدية، الجهاز الليمفاوي، الهرمونات الأنثوية",
                "diseases": "اضطرابات الهضم، عسر الطمث، احتجاز السوائل",
                "psych": "الأمان العاطفي، التقلبات المزاجية، القلق السيكوسوماتي، اضطرابات النوم"
            },
            "عطارد": {
                "ext": "اليدين، الذراعين، الكتفين، الأعضاء الحسية",
                "int": "الجهاز العصبي المحيطي، الرئتان، القصبة الهوائية، الأمعاء الدقيقة",
                "diseases": "أمراض التنفس، التشنجات العصبية، القولون العصبي",
                "psych": "التفكير الزائد (Overthinking)، القلق العصبي، إجهاد الدماغ والتركيز"
            },
            "الزهرة": {
                "ext": "الرقبة، الحلق، الفك السفلي، الظهر السفلي",
                "int": "الغدة الدرقية، الكلى، الدورة الدموية الوريدية، أسفل الظهر",
                "diseases": "أمراض الحلق، اضطرابات الكلى والترشيح، اضطرابات الأيض",
                "psych": "الانسجام الداخلي، العلاقات، الاستقرار العاطفي، الشراهة السيكوسوماتية"
            },
            "المريخ": {
                "ext": "الرأس، الوجه، العضلات",
                "int": "المخ، الدم، الحديد، الغدة النخامية، الجهاز التناسلي الذكري",
                "diseases": "الالتهابات الحادة، الحمى، الجروح والجراحات الطارئة",
                "psych": "الغضب المكبوت، التوتر الحاد، النزق، الاندفاعية النفسية"
            },
            "المشتري": {
                "ext": "الورك، الفخذين",
                "int": "الكبد، الدورة الدموية الشريانية، معالجة الدهون",
                "diseases": "أمراض الكبد، تضخم الأعضاء، السمنة، النقرس",
                "psych": "المبالغة، الإفراط في التفاؤل غير الواقعي، صدمات التوقعات"
            },
            "زحل": {
                "ext": "الجلد، الشعر، الأظافر، الركب",
                "int": "الهيكل العظمي، العظام، الأسنان، المفاصل، التكلس",
                "diseases": "أمراض العظام، التكلس، تصلب الشرايين، الأمراض المزمنة",
                "psych": "الاكتئاب، الخوف، الشعور بالعزلة، الضغط النفسي الطويل المدى"
            },
            "أورانوس": {
                "ext": "الساق، الكاحل",
                "int": "الجهاز العصبي المركزي، الإشارات الكهربائية للقلب والمخ",
                "diseases": "التشنجات المفاجئة، اضطرابات نبض القلب، الصدمات الكهربائية/العصبية",
                "psych": "الصدمات النفسية المفاجئة، الصراع مع القيود، التوتر العصبي الفجائي"
            },
            "نبتون": {
                "ext": "القدمين",
                "int": "الجهاز المناعي، غدد الصماء، الجهاز الليمفاوي",
                "diseases": "الحساسية، التسمم، ضعف المناعة، الأمراض الغامضة وصعبة التشخيص",
                "psych": "الهروبية، الاضطرابات النفسية الغامضة، الوسواس، الحساسية المفرطة"
            },
            "بلوتو": {
                "ext": "الأعضاء التناسلية، الشرج",
                "int": "الخلايا الجذعية، التجدد الخلوي، القولون، البروستات",
                "diseases": "الأورام الخبيثة، التغيرات الجينية، الأزمات الصحية الشديدة",
                "psych": "الترومات العميقة (Trauma)، الهوس الكبتي، تحولات الشخصية الجذرية"
            },
        }

        self.signs = {
            "الحمل": {"ext": "الرأس، الوجه", "int": "المخ، الغدة النخامية", "psych": "السرعة، التوتر الشديد، الصداع العصبي"},
            "الثور": {"ext": "الرقبة، الفك السفلي", "int": "الحلق، الغدة الدرقية", "psych": "العناد، الخوف من التغيير، التوتر الغذائي"},
            "الجوزاء": {"ext": "الكتفين، الذراعين", "int": "الرئتين، الجهاز العصبي", "psych": "تشتت الذهن، التفكير القلق، الإجهاد الهوائي"},
            "السرطان": {"ext": "الصدر، الثدي", "int": "المعدة، المريء، الرحم", "psych": "الحساسية المفرطة، تقلب المزاج، كبت المشاعر بالمعدة"},
            "الأسد": {"ext": "الظهر العلوي", "int": "القلب، العمود الفقري", "psych": "جرح الكبرياء، الإجهاد في إثبات الذات"},
            "العذراء": {"ext": "البطن", "int": "الأمعاء، الطحال، الجهاز الهضمي", "psych": "الوسواس الصحي، البحث عن الكمال، القولون العصبي"},
            "الميزان": {"ext": "أسفل الظهر", "int": "الكلى، الغدد الكظرية", "psych": "فقدان التوازن العاطفي، متردد، الإجهاد في إرضاء الآخرين"},
            "العقرب": {"ext": "الأعضاء التناسلية", "int": "القولون، الشرج، المثانة", "psych": "الشك، التكتم الشديد، الكبت العاطفي المتواجد عميقاً"},
            "القوس": {"ext": "الورك، الفخذين", "int": "الكبد، أسفل العمود الفقري", "psych": "نفاد الصبر، القلق من تقييد الحرية"},
            "الجدي": {"ext": "الركبتين، الجلد", "int": "العظام، الأسنان، المفاصل", "psych": "الشعور بالثقل، الجلد الصارم، الاكتئاب الجاد"},
            "الدلو": {"ext": "الساقين، الكاحل", "int": "الجهاز العصبي، الدورة الدموية", "psych": "الانفصال العاطفي، القلق من الاندماج الاجتماعي"},
            "الحوت": {"ext": "القدمين", "int": "الجهاز المناعي، الجهاز الليمفاوي", "psych": "الضعف أمام الضغوط، الحساسية الطاقية، القلق الإسقاطي"}
        }

        self.houses = {
            "البيت الأول": "الصحة العامة، البنية الجسدية، القوة الحيوية والمقاومة العامة.",
            "البيت الثاني": "التغذية، الموارد الغذائية، وطبيعة الاستهلاك الجسدي.",
            "البيت الثالث": "الجهاز التنفسي العلوي، الأعصاب المحيطية، والتواصل الذهني.",
            "البيت الرابع": "الوراثة الجينية، الصحة الباطنية، والسلام النفسي الداخلي.",
            "البيت الخامس": "طاقة القلب، الحيوية البدنية، والنشاط الترفيهي التشافي.",
            "البيت السادس": "بيت الأمراض الحادة، الروتين الصحي اليومي، والنظام الغذائي.",
            "البيت السابع": "التوازن الفسيولوجي والعلاقات العلاجية المساندة.",
            "البيت الثامن": "العمليات الجراحية، الأزمات الصحية الشديدة، والتحول الخلوي.",
            "البيت التاسع": "وظائف الامتصاص العلياء، الكبد، والمفاهيم الشفائية.",
            "البيت العاشر": "الهيكل العظمي، البنية الصلبة، واستجابة الجسد للضغط المهني.",
            "البيت الحادي عشر": "الدورة الدموية، الدعم النفسي الاجتماعي، والجهاز العصبي.",
            "البيت الثاني عشر": "الأمراض المزمنة، المستشفيات، العزلة النفسية، والصحة النفسية الباطنية."
        }

        self.asteroids = {
            "كايرون": "الجرح التشافي العميق؛ يشير إلى نقطة الضعف المزمنة التي تتطلب تقبلاً ورعاية ذاتية مستمرة.",
            "سيريس": "يحكم النمط الغذائي، اضطرابات الأكل، والربط بين الرعاية العاطفية وصحة الهضم.",
            "بالاس": "يمثل الذكاء الفسيولوجي، مناعة الجسم الاستراتيجية، والتناسق العصبي التشافي.",
            "جونو": "يرتبط بالتوازن الهرموني، صحة الحوض، وتأثير العلاقات الاجتماعية على الصحة الجسدية.",
            "فيستا": "يحكم الطاقة الحرارية للأيض، حيوية الشغف، ومخاطر الاستنزاف الشديد للطاقة (Burnout).",
            "هايجيا": "أيقونة الطب الوقائي؛ يحكم النظافة، الروتين الوقائي، الفحوصات الدورية ومنع تفاقم المرض."
        }

        self.aspects = {
            "الاقتران": "دمج مكثف للطاقات؛ قد يسبب تضخماً أو إجهاداً شديداً للأعضاء الممثلة.",
            "التسديس": "دعم فسيولوجي ونفسي إيجابي يُسهل الاستجابة للعلاج.",
            "التربيع": "صراع طاقي حاد يسبب التهابات، توتراً عصبياً، أو حبراً وظيفياً.",
            "التثليث": "انسجام تام وقدرة عالية على التشافي الذاتي واستعادة التوازن.",
            "المقابلة": "شد وجذب بين جهازين جسديين أو حالتين نفسيتين يسبب استنزافاً وظيفياً."
        }

    def analyze_placement(self, body: str, sign: str, house: str, is_asteroid: bool = False) -> str:
        if is_asteroid:
            body_info = self.asteroids.get(body, "غير متوفر")
            s_data = self.signs.get(sign, {})
            return (
                f"<b>تحليل الكويكب ({body}) في برج ({sign}) - ({house}):</b><br>"
                f"• <b>الدلالة الشفائية:</b> {body_info}<br>"
                f"• <b>المنطقة الجسدية:</b> {s_data.get('ext', '')} / {s_data.get('int', '')}.<br>"
                f"• <b>المجال النفسي:</b> {s_data.get('psych', '')}.<br>"
                f"• <b>سياق البيت الطبي:</b> {self.houses.get(house, '')}<br>"
            )

        p_data = self.planets.get(body, {})
        s_data = self.signs.get(sign, {})
        h_data = self.houses.get(house, "")

        is_medical_house = house in ["البيت الأول", "البيت السادس", "البيت الثامن", "البيت الثاني عشر"]
        warning_tag = ""

        if body == "القمر" and sign == "الجدي":
            warning_tag += "<br>🚨 <b>قاعدة خوان استاديا:</b> القمر في الجدي يدل على ضعف القوة الهضمية وحساسية الجهاز الهضمي للضغط النفسي."

        if is_medical_house:
            warning_tag += (
                f"<br>⚠️ <b>تنبيه طبي خاص:</b> تواجد {body} في {house} يجعل الأعضاء "
                f"التابعة له أكثر عرضة للاعتلال عند الإجهاد."
            )

        return (
            f"<b>التحليل التشريحي والنفسي لكوكب ({body}) في برج ({sign}) - ({house}):</b><br>"
            f"• <b>الأعضاء الجسدية:</b> {p_data.get('ext', '')} | {p_data.get('int', '')}.<br>"
            f"• <b>الاعتلالات المحتملة:</b> {p_data.get('diseases', '')}.<br>"
            f"• <b>المحفز النفسي السيكوسوماتي:</b> {p_data.get('psych', '')}.<br>"
            f"• <b>التأثير في البرج:</b> {s_data.get('ext', '')} - {s_data.get('int', '')} (نفسياً: {s_data.get('psych', '')}).<br>"
            f"• <b>تأثير البيت ({house}):</b> {h_data}{warning_tag}<br>"
        )

    def analyze_aspect(self, body1: str, body2: str, aspect_name: str, house1: str = "-", house2: str = "-",
                       is_body1_asteroid: bool = False, is_body2_asteroid: bool = False) -> str:
        p1 = self.planets.get(body1, {}) if not is_body1_asteroid else {}
        p2 = self.planets.get(body2, {}) if not is_body2_asteroid else {}
        aspect_meaning = self.aspects.get(aspect_name, "غير متوفر")

        rule_msg = ""
        bad_houses = ["البيت الأول", "البيت السادس", "البيت الثامن", "البيت الثاني عشر"]
        bad_aspects = ["الاقتران", "التربيع", "المقابلة"]

        if aspect_name in bad_aspects and (house1 in bad_houses or house2 in bad_houses):
            rule_msg = (
                f"<br>🚨 <b>قاعدة خوان استاديا:</b> اتصال غير متناغم ({aspect_name}) يرتبط ببيوت الصحة "
                f"يولد توتراً وظيفياً ونفسياً يطلب حماية الجسد من الإجهاد."
            )

        return (
            f"<b>التفاعل الفسيولوجي والتوتري ({aspect_name}) بين ({body1}) و ({body2}):</b><br>"
            f"• <b>المجال التأثيري لـ ({body1}):</b> {p1.get('int', 'كويكب وقائي/تشريحي')} (نفسياً: {p1.get('psych', 'تأثير طاقي')})<br>"
            f"• <b>المجال التأثيري لـ ({body2}):</b> {p2.get('int', 'كويكب وقائي/تشريحي')} (نفسياً: {p2.get('psych', 'تأثير طاقي')})<br>"
            f"• <b>طبيعة الاتصال:</b> {aspect_meaning}.{rule_msg}<br>"
        )

    def generate_medical_summary(self, placements: List[Dict[str, Any]], stelliums: Dict[str, int], aspects: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
        physical_status = []
        psych_status = []
        astro_advice = []
        prevention_plan = []

        if stelliums:
            for st_sign, count in stelliums.items():
                s_info = self.signs.get(st_sign, {})
                physical_status.append(
                    f"🔴 <b>ضغط فسيولوجي مُكثف (Stellium):</b> وجود {count} أجرام في برج {st_sign} يركّز الضغط الجسدي على: ({s_info.get('int', '')} / {s_info.get('ext', '')})."
                )
                psych_status.append(
                    f"🧠 <b>إجهاد نفسي مركّز:</b> التمركز في برج {st_sign} يُثير حالة من ({s_info.get('psych', '')}) التي قد تنعكس سيكوسوماتياً على صحتك."
                )

        medical_houses = ["البيت الأول", "البيت السادس", "البيت الثامن", "البيت الثاني عشر"]
        vulnerable_organs = []
        psych_triggers = []

        for p in placements:
            if p["house"] in medical_houses and not p["is_asteroid"]:
                planet_info = self.planets.get(p["body"], {})
                vulnerable_organs.append(planet_info.get('int', ''))
                if planet_info.get('psych'):
                    psych_triggers.append(f"{p['body']}: {planet_info.get('psych')}")

        if vulnerable_organs:
            unique_organs = list(set([o for o in vulnerable_organs if o]))[:4]
            physical_status.append(
                f"🟡 <b>الأعضاء الأكثر حساسية للضغط:</b> بناءً على بيوت الصحة والجراحة (1، 6، 8، 12)، يجدر إيلاء رعاية خاصة لـ: ({' ، '.join(unique_organs)})."
            )

        if psych_triggers:
            psych_status.append(
                f"🟡 <b>المحفزات النفسية للجسد:</b> لوحظ وجود مؤشرات للتوتر النفسي ترتبط بـ ({' | '.join(psych_triggers[:2])})."
            )

        for p in placements:
            b_name = p["body"]
            if b_name == "هايجيا":
                astro_advice.append("🛡️ <b>توصية هايجيا (Hygeia):</b> خط دفاعك الأول هو الطب الوقائي، النظافة الصحية، والفحوصات الدورية المستمرة.")
            elif b_name == "كايرون":
                astro_advice.append("🩹 <b>توصية كايرون (Chiron):</b> تقبّل نقاط الضعف البدنية وتجنب الإجهاد في علاجها؛ التشافي يكمن في الرعاية الذاتية المعتدلة.")
            elif b_name == "سيريس":
                astro_advice.append("🥗 <b>توصية سيريس (Ceres):</b> استقرارك الهضمي والجسدي مرتبط بنمط تغذية متوازن ورعاية عاطفية ملائمة.")
            elif b_name == "فيستا":
                astro_advice.append("🔥 <b>توصية فيستا (Vesta):</b> انتبه لمستويات الطاقة والأيض، وتجنب استنزاف طاقاتك حتى لا تصاب بالإرهاق (Burnout).")

        prevention_plan.append("✔️ **الفحوصات:** إجراء فحوصات الدم الدورية والمؤشرات الطبية الكلاسيكية تحت إشراف طبيب مختص.")
        prevention_plan.append("✔️ **النمط الغذائي:** تنظيم وجبات الطعام وتقليل المواد المسببة للالتهابات وخاصة عند التوتر.")
        prevention_plan.append("✔️ **الصحة النفسية:** ممارسة تمارين التنفس، الاسترخاء، والنوم الكافي لتخفيف تأثير التوتر السيكوسوماتي على الأعضاء الحيوية.")

        if not physical_status:
            physical_status.append("🟢 الخارطة لا تظهر تركيزات مرضية حادة بناءً على المعطيات الأساسية.")
        if not psych_status:
            psych_status.append("🟢 الحالة النفسية الاستدلالية تظهر توازناً طاقياً عاماً.")

        return physical_status, psych_status, astro_advice, prevention_plan


# ==========================================
# 3. واجهة تطبيق Streamlit والحسابات
# ==========================================
st.set_page_config(page_title="نظام التشخيص الفلكي الطبي", layout="wide")

DISCLAIMER_TEXT = (
    "🚨 **تحذير طبي هام :** هذا التقرير هو لأغراض البحث الفلكي الاستدلالي والاسترشادي فقط، "
    "ولا يغنيك بأي حال من الأحوال عن زيارة الطبيب المختص، إجراء الفحوصات الطبية المعترف بها طبياً، "
    "أو اتباع العلاجات المعتمدة."
)

st.error(DISCLAIMER_TEXT)

st.title("🏥 نظام التشخيص الطبي الفلكي من خلال هندسة الكواكب - من تاريخ الميلاد")
st.write(
    "نظام فلكي للتشخيص الطبي المبكر يستند على بحوث الكويكبات الشفائية الحديثة، "
    "لتقييم الحالة الجسدية والنفسية وإحداث نظام وقائي استباقي - برمجة الباحث الفلكي و الدكتور / ملهم احمد ."
)

BODIES = {
    "الشمس (Sun)": swe.SUN,
    "القمر (Moon)": swe.MOON,
    "عطارد (Mercury)": swe.MERCURY,
    "الزهرة (Venus)": swe.VENUS,
    "المريخ (Mars)": swe.MARS,
    "المشتري (Jupiter)": swe.JUPITER,
    "زحل (Saturn)": swe.SATURN,
    "أورانوس (Uranus)": swe.URANUS,
    "نبتون (Neptune)": swe.NEPTUNE,
    "بلوتو (Pluto)": swe.PLUTO,
    "راهو (North Node)": swe.TRUE_NODE,
    "ليليث (True Lilith)": swe.OSCU_APOG,
}

ASTEROIDS = {
    "كايرون (Chiron)": swe.CHIRON,
    "سيريس (Ceres)": swe.CERES,
    "بالاس (Pallas)": swe.PALLAS,
    "جونو (Juno)": swe.JUNO,
    "فيستا (Vesta)": swe.VESTA,
    "هايجيا (Hygeia)": 10010,
}

ZODIAC_SIGNS = [
    "الحمل", "الثور", "الجوزاء", "السرطان", "الأسد", "العذراء",
    "الميزان", "العقرب", "القوس", "الجدي", "الدلو", "الحوت"
]

HOUSE_ARABIC_NAMES = {
    1: "البيت الأول", 2: "البيت الثاني", 3: "البيت الثالث", 4: "البيت الرابع",
    5: "البيت الخامس", 6: "البيت السادس", 7: "البيت السابع", 8: "البيت الثامن",
    9: "البيت التاسع", 10: "البيت العاشر", 11: "البيت الحادي عشر", 12: "البيت الثاني عشر"
}


def get_zodiac_sign_and_degree(longitude: float) -> Tuple[str, int, float]:
    sign_index = int(longitude // 30)
    degree = longitude % 30
    return ZODIAC_SIGNS[sign_index], sign_index, degree


# شريط إدخال البيانات SideBar
with st.sidebar:
    st.header("بيانات صاحب الخارطة")
    name = st.text_input("الاسم:", "أحمد")
    gender = st.selectbox("الجنس:", ["ذكر", "أنثى"])
    dob = st.date_input("تاريخ الميلاد:", min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())

    selected_country = st.selectbox("الدولة:", country_list)
    country_code = country_map[selected_country]
    country_cities_data = cities_by_country.get(country_code, [])
    city_names = [city["name"] for city in country_cities_data]

    if not city_names:
        city_names = ["لا توجد بيانات مدن لهذا البلد"]

    selected_city_name = st.selectbox("المدينة:", city_names)
    submit = st.button("استخراج التقرير الطبي")

if submit and name and dob and selected_city_name != "لا توجد بيانات مدن لهذا البلد":
    city_data = next((city for city in country_cities_data if city["name"] == selected_city_name), None)

    if city_data:
        lat = city_data["latitude"]
        lng = city_data["longitude"]
        tz_str = tf.timezone_at(lng=lng, lat=lat) or "UTC"

        default_time = datetime.time(12, 0)
        jd = get_utc_julian_day(dob, default_time, tz_str)

        results = []
        sun_sign_index = 0
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED

        # 1. استخراج الكواكب
        for name_ar, planet_id in BODIES.items():
            try:
                pos, _ = swe.calc_ut(jd, planet_id, flags)
                longitude = pos[0]
                sign_name, sign_idx, degree = get_zodiac_sign_and_degree(longitude)

                if planet_id == swe.SUN:
                    sun_sign_index = sign_idx

                results.append({
                    "الكوكب": name_ar,
                    "البرج": sign_name,
                    "الدرجة": f"{degree:.2f}°",
                    "خط الطول": longitude,
                })
            except Exception as e:
                st.error(f"خطأ في حساب {name_ar}: {e}")

        # 2. حساب كيتو (South Node)
        try:
            rahu_lon = next(r["خط الطول"] for r in results if r["الكوكب"] == "راهو (North Node)")
            ketu_lon = (rahu_lon + 180) % 360
            ketu_sign, _, ketu_deg = get_zodiac_sign_and_degree(ketu_lon)
            results.append({
                "الكوكب": "كيتو (South Node)",
                "البرج": ketu_sign,
                "الدرجة": f"{ketu_deg:.2f}°",
                "خط الطول": ketu_lon,
            })
        except StopIteration:
            pass

        # 3. استخراج الكويكبات
        for name_ar, ast_id in ASTEROIDS.items():
            try:
                pos, _ = swe.calc_ut(jd, ast_id, flags)
                longitude = pos[0]
                sign_name, _, degree = get_zodiac_sign_and_degree(longitude)
                results.append({
                    "الكوكب": name_ar,
                    "البرج": sign_name,
                    "الدرجة": f"{degree:.2f}°",
                    "خط الطول": longitude,
                })
            except Exception:
                pass

        # 4. اسقاط البيوت الشمسية الطبية
        for r in results:
            planet_sign_index = ZODIAC_SIGNS.index(r["البرج"]) if r["البرج"] in ZODIAC_SIGNS else -1
            if planet_sign_index != -1:
                house_number = ((planet_sign_index - sun_sign_index) % 12) + 1
                r["البيت"] = HOUSE_ARABIC_NAMES.get(house_number, "-")
            else:
                r["البيت"] = "-"

        # 5. حساب الاتصالات
        aspects_results = []
        for body1, body2 in itertools.combinations(results, 2):
            lon1 = body1["خط الطول"]
            lon2 = body2["خط الطول"]
            aspect = get_aspect(lon1, lon2, orb=8)

            if aspect:
                aspects_results.append({
                    "الجرم الأول": body1["الكوكب"],
                    "بيت الأول": body1.get("البيت", "-"),
                    "الجرم الثاني": body2["الكوكب"],
                    "بيت الثاني": body2.get("البيت", "-"),
                    "الاتصال": aspect,
                })

        display_results = [{k: v for k, v in r.items() if k != "خط الطول"} for r in results]
        st.success(f"تم حساب البيانات الطبية والسيكوسوماتية بنجاح لـ: {name} ({gender}) - {selected_country}، {selected_city_name}")

        st.subheader("📊 المواقع الفلكية والبيوت الطبية")
        df_planets = pd.DataFrame(display_results)
        st.dataframe(df_planets, use_container_width=True)

        analyzer = MedicalAstroAnalyzer()

        # فحص التمركز الكوكبي
        sign_counts = {}
        for r in results:
            s = r["البرج"]
            sign_counts[s] = sign_counts.get(s, 0) + 1

        stelliums = {s: count for s, count in sign_counts.items() if count >= 3}
        if stelliums:
            for st_sign, count in stelliums.items():
                s_info = analyzer.signs.get(st_sign, {})
                st.warning(
                    f"⚠ **تمركز كوكبي (Stellium) في {st_sign}:** يتواجد {count} أجرام هنا، "
                    f"مما يخلق ضغطاً مكثفاً على: ({s_info.get('int', '')}) ونفسياً على: ({s_info.get('psych', '')})."
                )

        if aspects_results:
            st.subheader("📐 الاتصالات الفلكية المؤثرة فسيولوجياً")
            df_aspects = pd.DataFrame(aspects_results)
            st.dataframe(df_aspects, use_container_width=True)

        st.header("🩺 التقرير الطبي والتشخيص السيكوسوماتي")

        placements_for_summary = []

        st.subheader("أولاً: تحليل مواضع الكواكب والكويكبات")
        for r in results:
            full_name = r["الكوكب"]
            sign = r["البرج"]
            house = r["البيت"]
            body_ar = full_name.split(" (")[0]
            is_asteroid = body_ar in analyzer.asteroids

            placements_for_summary.append({
                "body": body_ar, "sign": sign, "house": house, "is_asteroid": is_asteroid
            })

            if (body_ar in analyzer.planets or is_asteroid) and house != "-":
                with st.expander(f"التشخيص الطبي والنفسي لـ {body_ar} في {sign} ({house})"):
                    report = analyzer.analyze_placement(body=body_ar, sign=sign, house=house, is_asteroid=is_asteroid)
                    st.markdown(report, unsafe_allow_html=True)

        st.subheader("ثانياً: تفاعلات الاتصالات والروابط العصبية")
        if aspects_results:
            for asp in aspects_results:
                body1_ar = asp["الجرم الأول"].split(" (")[0]
                body2_ar = asp["الجرم الثاني"].split(" (")[0]
                h1 = asp["بيت الأول"]
                h2 = asp["بيت الثاني"]
                aspect_name = asp["الاتصال"]

                is_ast1 = body1_ar in analyzer.asteroids
                is_ast2 = body2_ar in analyzer.asteroids

                if (body1_ar in analyzer.planets or is_ast1) and (body2_ar in analyzer.planets or is_ast2):
                    with st.expander(f"التفاعل الفسيولوجي لـ {aspect_name} بين {body1_ar} و {body2_ar}"):
                        report_asp = analyzer.analyze_aspect(
                            body1=body1_ar, body2=body2_ar, aspect_name=aspect_name,
                            house1=h1, house2=h2, is_body1_asteroid=is_ast1, is_body2_asteroid=is_ast2
                        )
                        st.markdown(report_asp, unsafe_allow_html=True)

        # ==========================================
        # 4. عرض الملخص الطبي للحالة الكاملة
        # ==========================================
        st.header("📝 الملخص الطبي للحالة وأهم النصائح")

        phys_sum, psych_sum, astro_adv, prev_plan = analyzer.generate_medical_summary(
            placements_for_summary, stelliums, aspects_results
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info("🩺 **1. التقييم والوضع الجسدي:**\n\n" + "\n\n".join(phys_sum))
            st.warning("🧠 **2. التقييم والوضع النفسي:**\n\n" + "\n\n".join(psych_sum))

        with col2:
            st.success("🪐 **3. أهم نصائح الكواكب والكويكبات:**\n\n" + ("\n\n".join(astro_adv) if astro_adv else "✔️ التوازن الطاقي للأجرام مستقر."))
            st.info("📋 **4. خطة الرعاية والوقاية المقترحة:**\n\n" + "\n\n".join(prev_plan))

        # ==========================================
        # 5. بناء التقرير المطبوع HTML / PDF الشامل والكامل
        # ==========================================
        # أ. إعداد قوائم الملخص
        html_phys = "".join([f"<li>{s}</li>" for s in phys_sum])
        html_psych = "".join([f"<li>{s}</li>" for s in psych_sum])
        html_adv = "".join([f"<li>{a}</li>" for a in astro_adv]) if astro_adv else "<li>لا توجد تحذيرات كوكبية حادة.</li>"
        html_prev = "".join([f"<li>{p}</li>" for p in prev_plan])

        # ب. إعداد جدول مواقع الكواكب
        html_planet_rows = ""
        for item in display_results:
            html_planet_rows += f"<tr><td><b>{item['الكوكب']}</b></td><td>{item['البرج']}</td><td>{item['الدرجة']}</td><td>{item['البيت']}</td></tr>"

        # ج. إعداد جدول الاتصالات الفلكية
        html_aspect_rows = ""
        if aspects_results:
            for asp in aspects_results:
                html_aspect_rows += f"<tr><td><b>{asp['الجرم الأول']}</b> ({asp['بيت الأول']})</td><td><span class='badge'>{asp['الاتصال']}</span></td><td><b>{asp['الجرم الثاني']}</b> ({asp['بيت الثاني']})</td></tr>"

        # د. إعداد التحليل التفصيلي لمواقع الأجرام
        html_placements_detail = ""
        for r in results:
            full_name = r["الكوكب"]
            sign = r["البرج"]
            house = r["البيت"]
            body_ar = full_name.split(" (")[0]
            is_asteroid = body_ar in analyzer.asteroids

            if (body_ar in analyzer.planets or is_asteroid) and house != "-":
                rep = analyzer.analyze_placement(body=body_ar, sign=sign, house=house, is_asteroid=is_asteroid)
                html_placements_detail += f"<div class='card detail-card'><h4>📍 {full_name} في برج {sign} ({house})</h4><p>{rep}</p></div>"

        # هـ. إعداد التحليل التفصيلي للاتصالات
        html_aspects_detail = ""
        if aspects_results:
            for asp in aspects_results:
                body1_ar = asp["الجرم الأول"].split(" (")[0]
                body2_ar = asp["الجرم الثاني"].split(" (")[0]
                h1 = asp["بيت الأول"]
                h2 = asp["بيت الثاني"]
                aspect_name = asp["الاتصال"]

                is_ast1 = body1_ar in analyzer.asteroids
                is_ast2 = body2_ar in analyzer.asteroids

                if (body1_ar in analyzer.planets or is_ast1) and (body2_ar in analyzer.planets or is_ast2):
                    rep_asp = analyzer.analyze_aspect(
                        body1=body1_ar, body2=body2_ar, aspect_name=aspect_name,
                        house1=h1, house2=h2, is_body1_asteroid=is_ast1, is_body2_asteroid=is_ast2
                    )
                    html_aspects_detail += f"<div class='card detail-card'><h4>🔗 اتصال {aspect_name} بين {asp['الجرم الأول']} و {asp['الجرم الثاني']}</h4><p>{rep_asp}</p></div>"

        # و. إعداد كتل التمركز الكوكبي (Stelliums)
        html_stelliums = ""
        if stelliums:
            for st_sign, count in stelliums.items():
                s_info = analyzer.signs.get(st_sign, {})
                html_stelliums += f"""
                <div class='warning-box' style='border-right-color: #f39c12; background-color: #fef9e7; color: #7d6608;'>
                    ⚠️ <b>تمركز كوكبي مكثف (Stellium) في برج {st_sign} ({count} أجرام):</b><br>
                    • <b>الضغط الفسيولوجي:</b> {s_info.get('int', '')} / {s_info.get('ext', '')}<br>
                    • <b>الضغط النفسي:</b> {s_info.get('psych', '')}
                </div>
                """

        # ز. بناء كود HTML النهائي القابل للطباعة
        html_content = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>التقرير الطبي الشامل - {name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #2c3e50; margin: 0; padding: 25px; line-height: 1.7; }}
        .report-container {{ max-width: 950px; margin: auto; background: #ffffff; padding: 35px; border: 1px solid #e1e8ed; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .warning-box {{ background-color: #fff3f3; color: #c0392b; padding: 18px; border-right: 6px solid #e74c3c; margin-bottom: 25px; font-weight: 500; border-radius: 6px; line-height: 1.6; }}
        .summary-section {{ background-color: #f8f9fa; padding: 22px; border: 1px solid #e9ecef; border-radius: 10px; margin-bottom: 30px; }}
        .card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
        .detail-card {{ border-right: 4px solid #3498db; }}
        h1 {{ text-align: center; color: #2c3e50; font-size: 26px; margin-bottom: 5px; font-weight: 700; }}
        h2 {{ color: #16a085; border-bottom: 2px solid #16a085; padding-bottom: 8px; margin-top: 35px; font-size: 20px; font-weight: 600; }}
        h3 {{ margin-top: 15px; font-size: 16px; font-weight: 600; }}
        h4 {{ margin: 0 0 10px 0; color: #2980b9; font-size: 16px; }}
        ul {{ padding-right: 20px; margin: 8px 0; }}
        li {{ margin-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 25px; background: #fff; }}
        th, td {{ padding: 10px 12px; text-align: center; border-bottom: 1px solid #e2e8f0; font-size: 13.5px; }}
        th {{ background-color: #34495e; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #fcfcfc; }}
        .badge {{ background-color: #e8f8f5; color: #117864; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .print-btn {{ display: block; width: 280px; margin: 35px auto 10px auto; padding: 14px; background-color: #27ae60; color: white; text-align: center; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 16px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); transition: background 0.2s; }}
        .print-btn:hover {{ background-color: #219150; }}
        @media print {{ 
            .print-btn {{ display: none !important; }} 
            body {{ background-color: #fff; padding: 0; }}
            .report-container {{ border: none; box-shadow: none; padding: 0; max-width: 100%; }}
            .card {{ page-break-inside: avoid; }}
            h2 {{ page-break-after: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="warning-box">
            ⚠️ <b>تحذير طبي هام:</b> {DISCLAIMER_TEXT}
        </div>
        
        <h1>🏥 التقرير الطبي والتشخيص الفلكي الشامل</h1>
        <p style="text-align:center; color: #7f8c8d; margin-top: 0; font-size: 15px;">
            <b>اسم المريض:</b> {name} | <b>الجنس:</b> {gender} | <b>تاريخ الميلاد:</b> {dob} | <b>الموقع:</b> {selected_country}، {selected_city_name}
        </p>

        <!-- 1. الملخص والتوصيات -->
        <div class="summary-section">
            <h2 style="margin-top:0; color:#2c3e50; border-bottom-color: #2c3e50;">📝 أولاً: الملخص الطبي التشخيصي</h2>
            
            <h3 style="color:#c0392b;">1. الوضع الجسدي والفسيولوجي:</h3>
            <ul>{html_phys}</ul>

            <h3 style="color:#d35400;">2. الوضع النفسي والسيكوسوماتي:</h3>
            <ul>{html_psych}</ul>

            <h3 style="color:#27ae60;">3. أهم نصائح الأجرام الفلكية:</h3>
            <ul>{html_adv}</ul>

            <h3 style="color:#2980b9;">4. خطة الوقاية والرعاية الموصى بها:</h3>
            <ul>{html_prev}</ul>
        </div>

        {html_stelliums}

        <!-- 2. جدول الأجرام والبيوت الطبية -->
        <h2>📊 ثانياً: المواقع الفلكية والبيوت الطبية</h2>
        <table>
            <thead>
                <tr>
                    <th>الكوكب / الكويكب</th>
                    <th>البرج</th>
                    <th>الدرجة</th>
                    <th>البيت الطبي المقابل</th>
                </tr>
            </thead>
            <tbody>
                {html_planet_rows}
            </tbody>
        </table>

        <!-- 3. جدول الاتصالات الفلكية -->
        {'<h2>📐 ثالثاً: الاتصالات الفلكية الفسيولوجية</h2><table><thead><tr><th>الجرم الأول (بيته)</th><th>نوع الاتصال</th><th>الجرم الثاني (بيته)</th></tr></thead><tbody>' + html_aspect_rows + '</tbody></table>' if html_aspect_rows else ''}

        <!-- 4. التحليل التفصيلي لمواقع الكواكب -->
        <h2>🩺 رابعاً: التحليل التشريحي والسيكوسوماتي التفصيلي لمواقع الكواكب</h2>
        {html_placements_detail}

        <!-- 5. التحليل التفصيلي للاتصالات -->
        {'<h2>🧠 خامساً: التحليل التفصيلي للروابط العصبية والاتصالات</h2>' + html_aspects_detail if html_aspects_detail else ''}

        <button class="print-btn" onclick="window.print()">🖨️ طباعة التقرير الطبي الكامل / حفظ PDF</button>
    </div>
</body>
</html>
"""

        # ==========================================
        # 6. المعاينة وتنزيل الملف
        # ==========================================
        st.markdown("---")
        st.subheader("📄 معاينة التقرير النهائي الشامل القابل للطباعة")
        components.html(html_content, height=850, scrolling=True)

        output_filename = "comprehensive_medical_astrology_report.html"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        st.download_button(
            label="📥 تحميل التقرير الطبي الكامل كملف HTML",
            data=html_content,
            file_name=f"Medical_Report_{name}.html",
            mime="text/html",
        )