"""Deterministic contact-field parsing (regex + heuristics, ZERO LLM cost).

Why not ask the LLM for the email? Because a 7B model hallucinates digits in
phone numbers and "cleans up" emails. Regex is exact, free, and -- crucially --
repairable: we can see *why* a match failed and fix it.

The hard part is OCR noise. Scanned resumes come out like:

    "J ohn  Doe"      "john.doe @gmail .com"     "555-555-5555-example@example.com"
    "(555)  5S5-1234"  "linkedln.com/in/johndoe" (that's a lowercase L, not an i)

So every matcher below runs twice: once on the raw text and once on a
"de-spaced" variant with all whitespace stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
_LOOKALIKES = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u00b7": "-", "\u25cf": "-",
    "\ufb01": "fi", "\ufb02": "fl", "\u00a0": " ",
}
# OCR digit/letter confusions we repair ONLY inside phone candidates.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"})

EMAIL_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+\-]{0,62}[A-Za-z0-9])?@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*\.[A-Za-z]{2,24}"
)
# Obfuscated forms: "john (at) gmail (dot) com", "john[at]gmail[dot]com"
OBFUSCATED_EMAIL_RE = re.compile(
    r"([A-Za-z0-9._%+\-]{2,64})\s*(?:\[|\(|\s)\s*(?:at|@)\s*(?:\]|\)|\s)\s*"
    r"([A-Za-z0-9.\-]{2,64})\s*(?:\[|\(|\s)\s*(?:dot|\.)\s*(?:\]|\)|\s)\s*([A-Za-z]{2,24})",
    re.I,
)

PLACEHOLDER_DOMAINS = {
    # Generic placeholders baked into resume templates.
    "example.com", "example.org", "example.net", "email.com", "domain.com",
    "sample.com", "test.com", "yourmail.com", "youremail.com", "mail.com",
    "company.com", "gmail.co", "abc.com", "xyz.com",
    # Template vendor addresses -- extremely common in downloaded samples.
    "qwikresume.com", "enhancv.com", "reallygreatsite.com", "novoresume.com",
    "resume.io", "zety.com", "live-career.uk", "visualcv.com", "myperfectresume.com",
    "hiration.com", "cakeresume.com", "kickresume.com", "jobscan.co",
}
TLD_OK = re.compile(r"^[a-z]{2,24}$")


def _levenshtein1(a: str, b: str) -> bool:
    """True if a and b differ by at most one edit (OCR often mangles 1 char)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) < len(b):
        a, b = b, a
    if len(a) == len(b):  # substitutions only
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    # one insertion/deletion
    for i in range(len(a)):
        if a[:i] + a[i + 1:] == b:
            return True
    return False


def is_placeholder_domain(domain: str) -> bool:
    d = domain.lower().strip()
    if d in PLACEHOLDER_DOMAINS:
        return True
    # Catch OCR damage such as "qwikresumc.com" (c read as e).
    return any(_levenshtein1(d, p) for p in PLACEHOLDER_DOMAINS if abs(len(p) - len(d)) <= 1)

# Phone: we deliberately do NOT use \b on the right-hand side, because OCR
# often glues the phone to the email (see the example above).
PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s.\-]?)?"                 # optional +CC
    r"(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])"  # area code
    r"\d{3}[\s.\-]?\d{2,4}(?:[\s.\-]?\d{2,4})?"
)
LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/([A-Za-z0-9\-_%]{3,60})", re.I
)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-_.]{2,60})", re.I)
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+){1,4}(?:/[^\s]*)?")

# Short role words: only rejected on word boundaries (a name can contain
# them as a substring -- "Cleaveland" contains "lead").
NAME_STOPWORDS = re.compile(
    r"\b(resume|curriculum|vitae|cv|profile|summary|objective|contact|information|"
    r"engineer|analyst|developer|manager|scientist|consultant|accountant|intern|"
    r"experience|education|skills|email|phone|address|linkedin|github|"
    r"senior|junior|entry|level|associate|assistant|head|lead|director|officer|"
    r"specialist|executive|administrator|coordinator|supervisor|fresher|"
    r"experienced|professional|certified|chartered|business|technical|"
    r"technologies|competencies|qualifications|strengths|languages|hobbies)\b",
    re.I,
)
# Longer role words are safe to reject as substrings, which is what catches
# OCR-glued headings like "TECHNICALSKILLS" / "WORKEXPERIENCE" / "CORESKILLS".
NAME_SUBSTRING_STOPWORDS = (
    "analyst", "engineer", "manager", "accountant", "consultant", "developer",
    "experience", "education", "skills", "summary", "objective", "profile",
    "business", "technical", "certification", "qualification", "competenc",
    "achievement", "reference", "languages", "interests", "objective",
    # Vendor / product names. Two-column resumes often open with a skills
    # column, so camelCase splitting turns "HubSpot" into "Hub Spot" -- which
    # otherwise looks exactly like a person's name.
    "salesforce", "hubspot", "tableau", "powerbi", "jira", "confluence",
    "quickbooks", "netsuite", "workday", "servicenow", "zendesk", "shopify",
    "wordpress", "drupal", "magento", "trello", "asana", "notion", "airtable",
    "snowflake", "databricks", "docker", "kubernetes", "jenkins", "terraform",
    "meistertask", "meister", "sharepoint", "outlook", "windows", "linux", "android", "javascript",
    "typescript", "postgresql", "mongodb", "elasticsearch", "splunk",
)

# Common given names, used to split OCR/all-caps glued names such as
# "ROBERTSMITH" -> "ROBERT SMITH" or "MONICABROWN" -> "MONICA BROWN".
FIRST_NAMES = {
    "aaron", "abby", "abdul", "abigail", "adam", "adrian", "adriana", "alan",
    "alberto", "alex", "alexander", "alexandra", "alice", "alicia", "alison",
    "allen", "alyssa", "amanda", "amber", "amelia", "amit", "amy", "ana",
    "anders", "andre", "andrea", "andrew", "angela", "anita", "anjali", "ann",
    "anna", "anne", "anthony", "antonio", "anurag", "april", "arjun", "arnold",
    "arthur", "ashley", "austin", "ava", "avery", "barbara", "barry", "beatrice",
    "ben", "benjamin", "bernard", "beth", "betty", "beverly", "bharat", "bill",
    "billy", "bob", "bobby", "brad", "bradley", "brandon", "brenda", "brendan",
    "brian", "bruce", "bryan", "caleb", "cameron", "camila", "carl", "carlos",
    "carol", "caroline", "carrie", "casey", "cassandra", "cate", "catherine",
    "cathy", "cecilia", "chad", "charles", "charlie", "charlotte", "chen",
    "cheryl", "chloe", "chris", "christian", "christina", "christine",
    "christopher", "cindy", "claire", "clara", "claudia", "clifford", "cody",
    "colin", "colleen", "connor", "craig", "crystal", "curtis", "cynthia",
    "dale", "damian", "dan", "dana", "daniel", "daniela", "danielle", "danny",
    "darius", "darlene", "darren", "dave", "david", "dawn", "dean", "debbie",
    "deborah", "deepak", "dennis", "derek", "desmond", "devin", "diana", "diane",
    "diego", "dinesh", "dominic", "don", "donald", "donna", "doris", "dorothy",
    "douglas", "duncan", "dylan", "ed", "eddie", "edgar", "edith", "edmund",
    "eduardo", "edward", "edwin", "eileen", "elaine", "eleanor", "elena",
    "eli", "elias", "elijah", "elisa", "elizabeth", "ella", "ellen", "emanuel",
    "emil", "emily", "emma", "emmanuel", "enrique", "eric", "erica", "erik",
    "erin", "ernest", "esther", "ethan", "eugene", "eva", "evan", "evelyn",
    "fabio", "faith", "farah", "fatima", "felicia", "felix", "fernando",
    "fiona", "florence", "frances", "francis", "francisco", "frank", "fred",
    "freddie", "frederick", "gabriel", "gabriela", "gary", "gavin", "gayle",
    "gene", "geoffrey", "george", "georgia", "gerald", "gilbert", "gina",
    "giovanni", "gloria", "gordon", "grace", "graham", "grant", "greg",
    "gregory", "guadalupe", "gustavo", "hannah", "harold", "harriet", "harry",
    "heather", "hector", "helen", "henry", "herbert", "hilary", "holly",
    "hugh", "hugo", "ian", "ibrahim", "imani", "indira", "irene", "iris",
    "irma", "isaac", "isabel", "isabela", "isabella", "isabelle", "ismael", "ivan",
    "ivy", "jack", "jackie", "jackson", "jacob", "jacqueline", "jaden",
    "jaime", "jake", "james", "jamie", "jan", "jane", "janet", "janice",
    "jared", "jasmine", "jason", "javier", "jay", "jayden", "jean", "jeanne",
    "jeff", "jeffrey", "jenna", "jennifer", "jenny", "jeremy", "jerome",
    "jerry", "jesse", "jessica", "jesus", "jill", "jim", "jimmy", "joan",
    "joanna", "joaquin", "joe", "joel", "john", "johnny", "jon", "jonathan",
    "jordan", "jorge", "jose", "joseph", "josephine", "joshua", "joy", "joyce",
    "juan", "juanita", "judith", "judy", "julia", "julian", "julie", "julio",
    "june", "justin", "kaitlyn", "kara", "karen", "karim", "karl", "kate",
    "katherine", "kathleen", "kathryn", "kathy", "katie", "katrina", "kayla",
    "keith", "kelly", "kelsey", "ken", "kenneth", "kevin", "khalid", "kim",
    "kimberly", "kirk", "kristen", "kristin", "kyle", "lance", "lara", "lars", "larry",
    "laura", "lauren", "laurence", "lawrence", "leah", "lee", "leo", "leon",
    "leonard", "leslie", "lewis", "liam", "lila", "lily", "linda", "lindsay",
    "lisa", "liz", "logan", "lois", "lola", "lorenzo", "loretta", "lorraine",
    "louis", "louise", "lucas", "lucia", "lucy", "luis", "luke", "luna",
    "lydia", "lynn", "mabel", "madeline", "madison", "maggie", "mahmoud",
    "malcolm", "mandy", "manuel", "marc", "marcia", "marcus", "margaret",
    "maria", "mariah", "marie", "marilyn", "mario", "marion", "marisa",
    "marissa", "mark", "marlene", "marsha", "martha", "martin", "marvin",
    "mary", "mason", "mateo", "mathew", "matt", "matthew", "maureen", "maurice",
    "maya", "megan", "meera", "melanie", "melissa", "melvin", "mercedes",
    "meredith", "mia", "micah", "michael", "micheal", "michelle", "miguel",
    "mike", "mildred", "miles", "milton", "mina", "miranda", "miriam",
    "mitchell", "mohamed", "mohammed", "molly", "monica", "morgan", "moses",
    "muhammad", "murray", "nadia", "nancy", "naomi", "natalie", "natasha",
    "nathan", "nathaniel", "neil", "nicholas", "nick", "nicole", "nina",
    "noah", "noel", "nora", "norma", "norman", "olga", "olive", "oliver",
    "olivia", "omar", "orlando", "oscar", "owen", "pablo", "paige", "pamela",
    "patricia", "patrick", "paul", "paula", "paulo", "pedro", "peggy",
    "penelope", "perry", "pete", "peter", "philip", "phillip", "phoebe",
    "phyllis", "pierre", "piotr", "preeti", "priya", "rachel", "rafael",
    "ralph", "ramon", "randall", "randy", "raul", "ravi", "ray", "raymond",
    "rebecca", "regina", "reginald", "rene", "renee", "ricardo", "richard",
    "rick", "ricky", "rita", "robert", "roberto", "robin", "rodney", "roger",
    "roland", "roman", "ron", "ronald", "ronnie", "rosa", "rose", "rosemary",
    "ross", "roy", "ruben", "rudolph", "russell", "ruth", "ryan", "sabrina",
    "sadie", "salvador", "sam", "samantha", "samir", "samuel", "sandra",
    "sandy", "sara", "sarah", "saul", "scott", "sean", "sebastian", "selena",
    "sergio", "seth", "shane", "shannon", "sharon", "shaun", "shawn", "sheila",
    "shelby", "sheldon", "sherry", "shirley", "sidney", "simon", "sofia",
    "sonia", "sophia", "sophie", "spencer", "stacy", "stanley", "stefan",
    "stella", "stephanie", "stephen", "steve", "steven", "stuart", "sue",
    "susan", "suzanne", "sydney", "sylvia", "tamara", "tammy", "tanya",
    "tara", "ted", "teresa", "terrence", "terry", "thelma", "theodore",
    "theresa", "thomas", "tiffany", "tim", "timothy", "tina", "toby", "todd",
    "tom", "tommy", "tony", "tracy", "travis", "trevor", "tricia", "tristan",
    "tyler", "tyrone", "valerie", "vanessa", "vera", "vernon", "veronica",
    "vicki", "victor", "victoria", "vincent", "viola", "violet", "virginia",
    "vivian", "wade", "walter", "wanda", "warren", "wayne", "wendy", "wesley",
    "wilbur", "william", "willie", "wilson", "xavier", "yolanda", "yvonne",
    "zachary", "zoe",
}


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Names that legitimately contain an internal capital -- don't split these.
_CAMEL_KEEP = re.compile(r"^(Mc|Mac|De|Van|Von|La|Le|Di|Da|Del|San|St|O')[A-Z]")


def split_glued_name(token: str) -> str:
    """Split an ALL-CAPS glued name using a common-given-name list.

    "ROBERTSMITH" -> "ROBERT SMITH",  "MONICABROWN" -> "MONICA BROWN".
    Tries the longest prefix first, then the longest suffix, so it prefers
    "ORLANDO CAMPA" over "O RLANDOCAMPA".
    """
    t = token.upper()
    if not t.isalpha() or not (7 <= len(t) <= 24):
        return token
    # Never split a token that already *is* a given name ("Jordan" -> not
    # "Jor dan"), and only ever split at a given-name PREFIX: names are
    # written given-then-family, so a suffix match is almost always wrong.
    if t.lower() in FIRST_NAMES:
        return token
    best = None
    for i in range(3, len(t) - 2):
        if t[:i].lower() in FIRST_NAMES and len(t) - i >= 3:
            best = i  # keep the longest given-name prefix
    if best is None:
        return token
    if token.isupper():
        return f"{t[:best]} {t[best:]}".title()
    return f"{token[:best]} {token[best:]}"


def split_camel(token: str) -> str:
    """'SamCrawford' -> 'Sam Crawford';  'StatenIsland' -> 'Staten Island'.

    OCR loves to delete the space between two capitalised words. Squashed
    keyword matching is immune to this, but the *displayed* name/location
    in the CSV looks broken, so we repair it there.
    """
    if " " in token or _CAMEL_KEEP.match(token):
        return token
    if re.search(r"[a-z][A-Z]", token):
        return _CAMEL.sub(" ", token)
    # All-caps or all-lower single token: try the given-name splitter.
    return split_glued_name(token)


@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""
    email_is_placeholder: bool = False
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""
    warnings: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    for bad, good in _LOOKALIKES.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------
def _valid_email(addr: str) -> bool:
    if not addr or addr.count("@") != 1:
        return False
    local, _, domain = addr.partition("@")
    if not (1 <= len(local) <= 64) or not (4 <= len(domain) <= 255):
        return False
    if ".." in local or local.startswith(".") or local.endswith("."):
        return False
    parts = domain.split(".")
    if len(parts) < 2 or not all(parts):
        return False
    return bool(TLD_OK.match(parts[-1].lower()))


def _scrub_email(addr: str) -> str:
    """Undo the most common OCR damage inside an email address."""
    addr = addr.strip().strip(".,;:|()<>[]{}\"'`")
    local, at, domain = addr.partition("@")
    # 'l'/'I' vs '1' and 'O' vs '0' are frequent, but only fix when the
    # character sits next to digits (heuristic: avoids mangling real names).
    def fix_local(s: str) -> str:
        out = []
        for i, ch in enumerate(s):
            nxt = s[i + 1] if i + 1 < len(s) else ""
            prv = s[i - 1] if i else ""
            if ch in "oOlISBZ" and (nxt.isdigit() or prv.isdigit()):
                out.append(ch.translate(_DIGIT_FIX))
            else:
                out.append(ch)
        return "".join(out)

    domain = re.sub(r"\s+", "", domain)
    # OCR often glues the phone onto the email:
    #   "555-555-5555-example@example.com"  ->  "example@example.com"
    m = re.match(r"^(\+?\d[\d\s().\-]{6,}[\s.\-])", local)
    if m and len(re.sub(r"\D", "", m.group(1))) >= 9:
        local = local[m.end():]
    return fix_local(local) + at + domain


def extract_email(text: str) -> tuple[str, bool]:
    """Return (email, is_placeholder). Tries clean text, then de-spaced text."""
    candidates: list[str] = []

    for m in EMAIL_RE.finditer(text):
        candidates.append(_scrub_email(m.group(0)))

    if not candidates:
        # "john.doe @ gmail . com" -> strip ALL whitespace and retry.
        flat = re.sub(r"\s+", "", text)
        for m in EMAIL_RE.finditer(flat):
            candidates.append(_scrub_email(m.group(0)))

    if not candidates:
        for m in OBFUSCATED_EMAIL_RE.finditer(text):
            cand = f"{m.group(1)}@{m.group(2)}.{m.group(3)}"
            candidates.append(_scrub_email(cand))

    seen: list[str] = []
    for c in candidates:
        if _valid_email(c) and c.lower() not in {s.lower() for s in seen}:
            seen.append(c)
    if not seen:
        return "", False

    # Prefer a non-placeholder domain; placeholder addresses (from resume
    # templates) are kept but flagged.
    for c in seen:
        if not is_placeholder_domain(c.split("@")[-1]):
            return c, False
    return seen[0], True


# --------------------------------------------------------------------------
# Phone
# --------------------------------------------------------------------------
def extract_phone(text: str, email: str = "") -> str:
    """Find the most plausible phone number.

    Emails are masked out first so a glued string like
    "555-555-5555-example@example.com" yields the phone, not a mess.
    """
    work = text
    if email:
        work = work.replace(email, " EMAILTOKEN ")

    best = ""
    best_score = -1
    for m in PHONE_RE.finditer(work):
        raw = m.group(0).strip(" -.,()")
        digits = re.sub(r"\D", "", raw)
        if not (9 <= len(digits) <= 15):
            continue
        # Reject pure date/ID-looking runs and ranges of years.
        if re.fullmatch(r"(19|20)\d{2}[-/]?(19|20)\d{2}", raw):
            continue

        score = 0
        if raw.strip().startswith("+"):
            score += 3
        # Indian mobile (10 digits starting 6-9) is the most common case here.
        if len(digits) == 10 and digits[0] in "6789":
            score += 4
        if len(digits) in (11, 12) and digits.startswith(("91", "1")):
            score += 3
        if re.search(r"[\-\s.()]", raw):       # grouped => intentional
            score += 2
        if len(digits) >= 12:
            score -= 2                          # probably an ID, not a phone
        # Penalise candidates glued to letters (e.g. an invoice number).
        ctx_start = max(0, m.start() - 2)
        ctx_end = min(len(work), m.end() + 2)
        if re.search(r"[A-Za-z]", work[ctx_start:m.start()]) and not raw.startswith("+"):
            score -= 3
        if re.search(r"[A-Za-z]", work[m.end():ctx_end]) and re.search(r"[A-Za-z]", raw):
            score -= 3
        if score > best_score:
            best_score, best = score, raw

    if not best:
        # De-spaced retry: "(555) 5S5 1234" -> "(555)5551234".
        flat = re.sub(r"\s+", "", work)
        for m in PHONE_RE.finditer(flat):
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            fixed = re.sub(r"[A-Za-z]", lambda c: c.group(0).translate(_DIGIT_FIX), raw)
            digits_fixed = re.sub(r"\D", "", fixed)
            if 9 <= len(digits_fixed) <= 15 and len(digits_fixed) >= len(digits):
                return raw if not re.search(r"[A-Za-z]", raw) else fixed
    return best


# --------------------------------------------------------------------------
# Social / location
# --------------------------------------------------------------------------
def extract_socials(text: str) -> tuple[str, str]:
    linkedin, github = "", ""
    m = LINKEDIN_RE.search(text)
    if m:
        linkedin = f"linkedin.com/in/{m.group(1).strip('/')}"
    else:
        # OCR loves turning linkedin's 'i' into 'l': "linkedln.com/in/foo"
        m2 = re.search(r"linked[lI]n\.com/(?:in|pub)/([A-Za-z0-9\-_%]{3,60})", text, re.I)
        if m2:
            linkedin = f"linkedin.com/in/{m2.group(1).strip('/')}"
    m = GITHUB_RE.search(text)
    if m:
        github = f"github.com/{m.group(1).strip('/')}"
    else:
        m2 = re.search(r"g[il]thub\.com/([A-Za-z0-9\-_.]{2,60})", text, re.I)
        if m2:
            github = f"github.com/{m2.group(1).strip('/')}"
    return linkedin, github


_CITY_LINE = re.compile(
    r"^([A-Z][A-Za-z.\-]+(?: [A-Z][A-Za-z.\-]+){0,3}),?\s*"
    r"([A-Z]{2}|[A-Z][A-Za-z ]{2,25})\s*,?\s*(\d{4,6})?$"
)

# A loose "Word Word" pattern is NOT enough to call something a location --
# "Sam Crawford" would match. Require an explicit comma plus a recognised
# region token (US state, common country) or a postal code.
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}
_COUNTRIES = {
    "usa", "u.s.a", "us", "united states", "united states of america", "uk",
    "united kingdom", "canada", "india", "australia", "germany", "france",
    "netherlands", "ireland", "singapore", "uae", "united arab emirates",
    "pakistan", "bangladesh", "philippines", "new zealand", "south africa",
    "spain", "italy", "switzerland", "sweden", "poland", "brazil", "mexico",
    "china", "japan", "nigeria", "kenya", "malaysia", "indonesia",
}
_ZIP_TAIL = re.compile(r"^[A-Za-z ]{2,25}\s*,?\s*\d{4,6}$")


def is_location_line(s: str) -> bool:
    """True for 'Dallas, TX', 'Edmondbury, United States', 'Pune, MH 411001'."""
    if "," not in s:
        return False
    tail = s.rsplit(",", 1)[1].strip()
    if not tail:
        return False
    if re.fullmatch(r"[A-Z]{2}", tail):
        return tail in _US_STATES
    if _ZIP_TAIL.match(tail):
        return True
    low = tail.lower().strip(".")
    return low in _US_STATE_NAMES or low in _COUNTRIES


def extract_location(lines: list[str]) -> str:
    """Look for a 'City, ST 12345' / 'City, Country' pattern in the header."""
    for line in lines[:12]:
        s = line.strip().strip("|·,-")
        if not (4 <= len(s) <= 60) or "@" in s or re.search(r"\d{3}[-.\s]\d{3}", s):
            continue
        if is_location_line(s):
            m = re.match(r"^([A-Za-z.\- ]+),\s*(.+)$", s)
            if m:
                city = " ".join(split_camel(w) for w in m.group(1).strip().split())
                # Drop a stray OCR bullet that landed before the city ("Q Edmondbury").
                city = re.sub(r"^[A-Za-z](?=\s+[A-Z][a-z])", "", city).strip(" .-")
                region = m.group(2).strip()
                # "NY10301" -> "NY 10301"
                region = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", region)
                return f"{city}, {region}"
    return ""


# --------------------------------------------------------------------------
# Name
# --------------------------------------------------------------------------
def extract_name(lines: list[str], fallback: str = "") -> tuple[str, str]:
    """First plausible title-case line in the header = the candidate's name.

    Returns (name, source) where source is 'header' or 'filename' (a
    filename fallback means we failed and the CSV should say so).
    """
    for line in lines[:8]:
        s = line.strip().strip("|·,-–—")
        if not (3 <= len(s) <= 60):
            continue
        if "@" in s or re.search(r"\d{3}", s):
            continue
        # Repair OCR-glued words FIRST: "ROBERTSMITH" -> "ROBERT SMITH" and
        # "EntryLevelBusinessAnalyst" -> "Entry Level Business Analyst", so the
        # role-word filter below can actually see the word "Analyst".
        s = " ".join(split_camel(w) for w in s.split())
        # Job titles and section headings are the #1 false positive here.
        if NAME_STOPWORDS.search(s):
            continue
        squashed_s = re.sub(r"[^a-z]+", "", s.lower())
        if any(w in squashed_s for w in NAME_SUBSTRING_STOPWORDS):
            continue
        # "Dallas, TX" / "Pittsburgh, PA" -- a location, not a person.
        if is_location_line(s):
            continue
        words = [w for w in re.split(r"\s+", s) if w]
        # Two-line names ("LARS" / "PETERS") are common in designed templates.
        idx = lines.index(line) if line in lines else -1
        # Only join a two-line name when it starts the document; otherwise a
        # skills list ("Salesforce" / "HubSpot") looks exactly like one.
        if len(words) == 1 and idx == 0 and len(lines) > 1:
            nxt = lines[idx + 1].strip().strip("|·,-–—")
            nxt_words = nxt.split()
            if (len(nxt_words) == 1 and nxt[:1].isupper() and nxt.isalpha()
                    and not NAME_STOPWORDS.search(nxt)
                    and not any(w in nxt.lower() for w in NAME_SUBSTRING_STOPWORDS)):
                s = f"{s} {nxt}"
                words = [s.split()[0], nxt]
        if not (1 <= len(words) <= 5):
            continue
        # A single-token name is only credible as the very first line.
        if len(words) == 1 and idx != 0:
            continue
        # Real names live in the first few lines. Anything deeper that also
        # contains no recognised given name is almost certainly a stray skill
        # or heading (e.g. "Google Applications" from a skills column).
        if idx > 2 and not any(w.strip(".,").lower() in FIRST_NAMES for w in words):
            continue
        # Require mostly Titlecase/UPPERCASE letters (names are capitalised).
        cap = sum(1 for w in words if w[:1].isupper())
        if cap < max(1, len(words) - 0):
            continue
        if sum(c.isalpha() or c in " .'-" for c in s) / max(1, len(s)) < 0.75:
            continue
        return s, "header"
    if fallback:
        clean = re.sub(r"[_\-]+", " ", fallback)
        clean = re.sub(r"\.(pdf|docx?|txt)$", "", clean, flags=re.I)
        return clean.strip().title(), "filename"
    return "", "none"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def parse_contacts(text: str, filename: str = "") -> ContactInfo:
    text = normalize(text or "")
    info = ContactInfo()

    email, placeholder = extract_email(text)
    info.email, info.email_is_placeholder = email, placeholder

    info.phone = extract_phone(text, email)
    info.linkedin, info.github = extract_socials(text)

    head = text[:1200]
    lines = [ln for ln in (l.strip() for l in head.splitlines()) if ln]
    info.name, src = extract_name(lines, filename)
    if src != "header":
        info.warnings.append(f"name-from-{src}")
    elif not any(w.strip(".,").lower() in FIRST_NAMES for w in info.name.split()):
        # We found a header line that looks like a name, but none of its words
        # is a recognised given name -- often a stray skill or section heading.
        info.warnings.append("name-low-confidence")
    info.location = extract_location(lines)

    if not info.email:
        info.warnings.append("no-email-found")
    if not info.phone:
        info.warnings.append("no-phone-found")
    if placeholder:
        info.warnings.append("placeholder-email")
    return info
