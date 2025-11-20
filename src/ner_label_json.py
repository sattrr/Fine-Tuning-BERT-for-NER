from label_studio_sdk import Client
import re
import os
import urllib.parse
import requests 
import json

# Config
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "79349c4580df4d083bf294ec9cb005f24965323d"
PROJECT_ID = 10
BASE_PATH = r"C:\Users\Satriock\Documents\Code\Dataset\labelstudiodata" 

DATE_PATTERN = re.compile(
    r"""
    (?<!\d)(
        # Format numerik
        \d{1,2}\s*[/-]\s*\d{1,2}(?:\s*[/-]\s*\d{2,4})?

        # Format dengan nama bulan
        | \d{1,2}(?:\s*[-–]\s*\d{1,2})?\s*
          (?:Jan(?:uari)?|Feb(?:ruari)?|Mar(?:et)?|Apr(?:il)?|Mei|
            Jun(?:i)?|Jul(?:i)?|Agu(?:stus)?|Sep(?:tember)?|
            Okt(?:ober)?|Nov(?:ember)?|Des(?:ember)?)
          (?:\s*\d{2,4})?

        # Tahun
        | (?:19|20)\d{2}
    )(?!\d)
    """,
    re.IGNORECASE | re.VERBOSE
)

RULES = {
    "NAME": ["rohis inggrit aprilia", "dewi retno sariwulan", "yosfi rahmi", "yosff rahmi", "an. s", "an. g", "an.g", "Gustina Berta Uli", "Ayu Dyah Kartika A",
             "an. d","an. aa", "Olivia Anggraeny", "Ratih Puspitaningtyas", "an. e", "Gelora Mangalik", "Noviyani", "an. k", "Ayuningtyas Dian A.", "An. INR",
             "Ratih Puspitasari", "an. a", "Ufairoh Maliha Shofwah", "An. AZ", "An. AS", "An. FT", "An. MRA", "An. ADP", "By. Ny. S", "An. IS",
             "An AAP", "Pratama", "An. Ahmad Syailendra", "Ny. NFA", "An. NFA", "An. DR", "ayuningtyas dian ariestiningsih", "An. FB", "An. MAAG",  
             "An. ZA", "An MU", "An. MU", "Tn. S", "An. Ka", "Ny. Ka", "An. NI", "An. Nu", "Matana", "Umboh", "Anggun Rindang C.", 
             "An. MK", "An. Alr", "An. ANS", "An. NL", "Aliviameita", "An. MAK", "BY. Ny.AN", "Ayunintyas Dian A.", "Bayi Ny.AS", "Ayuningtyas Dian A., S.Gz, MP",
             "Bayi MA", "Ny, F","An. ARF", "An SW", "An. SW", "Anggun Rindang Cempaka, S. Gz, M.S", "Dewi Retno Sariwulan, S.ST, M.Gizi", "Yosfi Rahmi, S.Gz, M.Sc", 
             "Prof. Dian Handayani, S.Gz, M.Sc, Ph.D", "Dr. Nurul Muslihah, SP, M.kes", "Dewi Retno Sariwulan, S.ST", "Ridna Tri Widyaningrum, S .Gz, MPH, RD",
             "Ayuningtyas Dian Ariestiningsih, S.Gz M.P", "Yosff Rahmi, S.Gz_, MSc", "Prof Dian Handayani, S.K.M, M.Kes., Ph.D.", "Yosfi Rahmi, S.Gz., M.Sc.",
             "Ayuningtyas Dian Ariestiningsih, S.Gz M.P.", "Ridna Tri Widyaningrum, S.Gz, MPH, RD.", "Ir. Erny Sulistyaningsih, M.MTr.", "Drs. Prasetyo",
             "Anggun Rindang Cempaka, S.Gz, Dietisien, M.S", "Ernida, S.ST", "Olivia Anggraeny, S.Gz, M.Biomed", "Ratih Puspitaningtyas, S.Gz. RD", 
             "Ratih Puspitaningtyas, S.Gz.", "Olivia Anggraeny, S.Gz.. M.Biomed", "Ayuningtyas Dian A., S.Gz., M.P", "Prof. Dian Handayani, SKM, M.Kes, PhD",
             "RAtih Puspitasari, S.Gz., RD", "Olivia Anggraeny, S.Gz., M.Biomed", "Anggun Rindang Cempaka, S.Gz, MS, RD", "Anindita Chandraintan Prathami, S.GZ, RD",
             "Ayu Dyah Kartika A, Sst. Gz", "Ayuningtyas Dian Ariestiningsih, S.Gz,, MP", "Ratih Puspitaningtyas, S.Gz, RD", "Ermida, SST", "Juwariyah, S.Gz., RD",
             "Ayuningtyas Dian Ariestiningsih, S.Gz, M.P", "Suci Purwati, S.Gz, RD", "Brelian Titihwara, RD.", "Pudji Astutik, S.KM., M.Kes", "Winarsih, S.Gz",
             "Dr. Nurul Muslihah, S.P., M.kes.", "Ayuningtyas Dian Ariestiningsih, S.Gz., MP", "Anggun Rindang Cempaka, S.Gz, M.S, Dietisien",
             "Anggun Rindang C., S. Gz., MS., Dietisien", "Dewi Retno S., SST. M.Gizi", "Pudji Astutik, SKM., M.Kes", "Ayuningtyas Dian A., S.Gz., MP",
             "Ayuningtyas Dian A., S.Gz., M.P", "Anggun Rindang C., S.Gz, Dietisien, M.S", "Endah Setyo Rahayu, DCN", "Ida Restyani, SST., M.KES., RD",
             "Yosfi Rahmi, S.Gz., M.Sc", "Frida Afiani., SST", "Galuh Ardianti., S.Gz", "dr. Mochamad Hafidin Ilham, Sp.An.", "Wilujeng Suci Untarsih, SST.",
             "Galuh Ardianti., S.Gz", "Ayuningtyas Dian Ariestiningsih, S.Gz, M.P.", "Catur Saptaning Wilujeng, S.Gz., M.P.H.", "Ermida, S.ST",
             "Prof. Dian Handayani, S.K.M, M.Kes, Ph.D.", "Ratih Puspitaningtyas, S.Gz.. RD", "Suci Purwati, S. Gz, RD", "Juwariyah, S.Gz, RD", 
             "Prof. Dian Handayani, S.K.M., M.Kes., Ph.D.", "Ayunintyas Dian A., S.Gz., M.P"],
    "DATE": DATE_PATTERN,
    "ADDR": ["kabupaten muna", "Jl. Embong Klojen", "Jl. Terusan Agus Salim", "Wajak Malang", "pasuruan", "malang", "Jatimulya, Jati guwi, Sumberpucung Kab. Malang",
             "Perum Permata Jingga C17 10/6", "Ploso Wonoayu", "Dukuh Jarsono 2/17 RT 03/RW 01 Wiyung Surabaya", "Dusun Turus RT.03/RW.05 Banjarejo, Ngantang, Malang",
             "Jl. Veteran 175 Blitar", "Gedangan, Kabupaten Malang", "Dinsos Kabupaten Sidoarjo", "Sidokepung RT 5/ RW 1, Buduran", "Jl. KH Malik, Malang",
             "Jalan Sawojajar, Kota Malang", "Jl Danau Limboto, Malang"],
    "ID": ["220170200111018", "2015038902022001", "197011131994032003", "19791203 200604 2002", "115740xxx", "123041xxx", "210070100111009",
           "197311141996032001", "2016068602042001", "922407xxx", "210070100111014", "232xxx", "231xxx", "210070100111022", "197007051994032011",
           "xxx794", "20140487 0605 2001", "198804122011012018", "11466731", "210070100111012", "1146xXxXXxX", "210070100111008", "1042xxxx",
           "114xxxxx", "210070100111004", "2013098902082001", "210070100111018", "1153****", "2143310", "210070100111011", "11461xxxx", 
           "1149xxxx", "68.74.49", "210070100111019", "165xx2", "1136xxxx", "19740128 200012 2 004", "19791203 200604 2 002", "xxxxx869",
           "112782xx", "11537xxx", "210070100111010", "220170200111022", "12304XXX", "210070100111015", "220170200111001", "19690405 199503 2 003",
           "19700125 199603 200 4", "1527xxx", "220170200111021", "9483XXX", "230170200111001", "74-75-xx", "220170200111005", "11578xxx",
           "220170200111013", "12219xxx", "220170200111017", "196904051995032003", "2206xxx", "19641112 198801 2 002", "197106031995032001",
           "11572xxx", "2014048706052001", "684xxx5", "20160686 0204 2 001", "684xxx1", "210070100111020", "1986020407120033", "114727xx",
           "220170200111004", "399xxx", "220170200111023", "00-13-7xxx", "1519xx", "197912032006042002", "538/b", "1970009251991022001",
           "19741011 200003 2 002"],
    "PII": ["islam", "kristen", "katolik", "hindu", "buddha", "konghucu", 
            r"\b08\d{8,12}\b", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "a", "b", "ab", "o", "golongan darah", "laki-laki", "perempuan", "menikah", "belum menikah"],
    "NUTR": ["vitamin", "vitamin a", "vitamin c", "vitamin k", "vitamin e", "vitamin b6", "vitamin b12", "vitamin d","vitamin bc", "vitamin b", 
             "vitamin c1x", "zinc", "as folat", "asam folat", "vit c", "vit a", "vit e", "fe", "zat besi", "protein", "karbohidrat", 
             "lemak", "omega 3", "kalsium", "natrium", "iodium", "klorida", "zink", "serat", "zn", "carbohydr", "kalium", "Proteinz",
             "vita", "vitc", "Vit. B12", "Vit. K", "Vit. C", "Vit. A", "Vit. D", "Vit. E", "Vit. B1", "As. Folat", "magnesium", "fosfat",
             "omega-3", "omega-6", "glukosa", "Lactate", "laktosa", "potasium"],
    "FOOD": ["susu", "tempe", "nasi", "tahu", "tempe", "sayur", "kentang", "ikan", "labu", "ikan tongkol", "ikan pindang", 
             "wortel", "buncis", "bayam", "pisang", "semangka", "telur", "jeruk", "biscuits", "daging sapi", "ayam", "ubi jalar",
             "ketela pohon", "bebek", "kerang", "ikan mujaer", "kacang kedelai", "kacang merah", "kacang tanah",
             "kacang hijau", "daun pepaya", "daun singkong", "daun tales", "kangkung", "sawi", "tomat", "durian", "jambu biji",
             "kedondong", "mangga", "nanas", "pepaya", "rambutan", "pisang raja", "apel", "pear", "jagung", "yoghurt", "teh", 
             "kare ayam", "yupi", "coklat", "permen", "sop", "sayuran", "biskuit regal", "wafer", "ikan lele", "roti tawar", 
             "Carrot", "Broccoli", "biscuit regal", "bubur kacang hijau", "coconut milk", "jamur kuping", "roti", "makaroni",
             "melon", "Ayam goreng", "daging ayam", "Sup kacang merah", "Ongseng", "gambas", "telur puyuh","macaroni",
             "oyong mentah", "pisang hijau", "ultra milk", "ikan kakap", "rawon", "soto", "manisah", "tempe bacem", "bakwan",
             "bakmie", "mie", "air mineral", "rolade", "sari kacang hijau", "grilled tori teriyaki", "roti kukus", "kembang kool",
             "nasi tim", "bakso", "pentol", "singkong", "ikan mujair", "sup", "udang", "kepiting", "papaya", "kopi", "soda", 
             "alpukat", "kol", "brokoli", "kacang panjang", "cilok", "agar agar", "cokelat", "gorengan", "biskuit", "jamur", 
             "peas", "lodeh", "mendol", "koloke", "empal", "tempe mendoan", "perkedel", "gurami", "jus jambu", "sayur asem", 
             "Bubur Instan", "pisang ambon", "daging", "semur", "madu", "snack", "ikan tuna", "timun", "puding", "egg roll", 
             "tofu", "jus", "ikan bandeng", "ikan patin", "kelengkeng", "ubi", "kangkong"],
    "COND": ["anemia", "diabetes", "diabetes mellitus", "marasmus", "kwashiorkor", "hipotrofi", "atrofi", "diare", "cacingan", 
             "anoreksia", "pneumonia", "dehidrasi", "demam", "diet", "batuk", "pilek", "odem", "asma", "mual", "muntah", "edema", 
             "asites", "inflamasi", "faringitis", "limfadenitis", "ruam", "gastroenteritis", "imunodefisiensi", "malnutrisi",
             "infeksi enteral", "infeksi virus", "infeksi parasit", "infeksi panteral", "tromboflebitis", "pusing", "alergi", 
             "bronkitis", "sinusitis", "otitis media", "laringitis", "kejang", "peritonitis", "ISPA", "ileus obstruksi","obesitas",
             "hipertensi", "dislipidemia", "obes", "koma", "hipertensi ensefalopati", "Impetigo", "skabies", "vertigo", "hipertensi aritmia",
             "migrain", "Leukositosis", "overweight", "Tuberculosis", "diabetes militus", "tbc", "Abses paru", "iritasi", "syok", "cacat",
             "trauma", "infeksi", "cerebral palsy", "epilepsi", "epilepsy", "hydrocephalus", "stunting", "kram", "keram", "Ascites", 
             "leukimia", "Hidrosefalus", "eritema", "kanker", "stress", "combustio", "Pharyngitis", "atresia ani", "premature", "prematur"]
}

FROM_NAME = "label"
TO_NAME = "text"

client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
project = client.get_project(PROJECT_ID)
print(f"Connected: {project.title} (ID={project.id})")

tasks = list(project.get_tasks())
print(f"{len(tasks)} task found\n")

total_entities = 0

for task in tasks:
    task_data = task.get("data", {})
    task_id = task['id']

    raw_text = task_data.get("text")
    
    if not raw_text:
        print(f"Skip task {task_id} (data 'text' is empty)")
        continue
    
    text_cleaned = re.sub(r'\\', '', raw_text)
    
    text_normalized = re.sub(r'[\r\n\t]+', ' ', text_cleaned)
    text = re.sub(r' {2,}', ' ', text_normalized).strip()

    if not text.strip():
        print(f"Task {task_id}: empty text")
        continue

    results = []

    for label, terms in RULES.items():
        if isinstance(terms, re.Pattern):
            pattern = terms
            matches = pattern.finditer(text)
        else:
            sorted_terms = sorted(terms, key=len, reverse=True)
            pattern = r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            results.append({
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "labels",
                "value": {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "labels": [label]
                }
            })
    
    num = len(results)
    total_entities += num

    if results:
        try:
            project.create_annotation(task_id, result=results)
            print(f"Task {task_id}: {num} entities labeled")

        except Exception as e:
            print(f"Failed to label task {task_id}: {e}")
    else:
        print(f"Task {task_id}: not found")

print("\n============================")
print(f"{total_entities} entities labeled")