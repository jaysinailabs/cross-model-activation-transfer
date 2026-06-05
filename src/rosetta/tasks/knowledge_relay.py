"""
Knowledge relay task data generator.

Generates factual passage + QA pairs.  Each passage is a structured
factual article (200-350 words) about a real topic.  Questions require
a specific numeric fact, named entity, or date from the passage.
Generation is deterministic given a seed; no network access required.

Task design (experiment guide §4.4, Task 3):
    Model A reads the full passage.
    Model B receives the relay (from Model A) + question, and must answer
    using only the relayed information.
    The task exposes how much factual detail is preserved in the relay.

Data format per sample:
    {
        "passage":     "<200-350 word factual article>",
        "question":    "<question answerable from passage>",
        "answer":      "<short string, verbatim from passage>",
        "passage_id":  <int, index into _CORPUS>,
        "relay_point": (
            "Model A reads passage; Model B receives relay and answers "
            "question using only the relay."
        )
    }

Split strategy:
    Splitting MUST be done at the passage level (all QA pairs from one
    passage go to the same split).  Mixing QA pairs from the same passage
    across train/val/test would cause data leakage: the translation layer
    trained on train would see the exact passage texts that appear in test.
    Use the passage_id field to group samples by passage before splitting.
    See m1_data_generation.split_knowledge_relay() for the reference impl.
"""

from __future__ import annotations

import random
from typing import Any

RELAY_POINT = (
    "Model A reads passage; Model B receives relay and answers question using only the relay."
)

# ---------------------------------------------------------------------------
# Passage + QA corpus
# Each entry: {"passage": str, "qas": [(question, answer), ...]}
# Target passage length: 200-350 words.
# ---------------------------------------------------------------------------

_CORPUS = [
    # --- ANIMALS -------------------------------------------------------
    {
        "passage": (
            "The African elephant (Loxodonta africana) is the largest land animal on Earth. "
            "Adult males, called bulls, typically weigh between 4,000 and 7,000 kilograms "
            "and stand 3 to 4 metres tall at the shoulder. Females are noticeably smaller, "
            "usually weighing between 2,700 and 3,600 kilograms. "
            "African elephants are found across sub-Saharan Africa, inhabiting savannahs, "
            "forests, and deserts. They are highly social animals, living in herds led by "
            "the oldest female, known as the matriarch. A herd typically consists of ten to "
            "twenty individuals, all closely related to the matriarch. "
            "Elephants have a gestation period of approximately 22 months, the longest of "
            "any land mammal. Calves weigh around 120 kilograms at birth and can walk within "
            "hours. Juvenile elephants nurse for up to four years. "
            "They are herbivores, consuming up to 300 kilograms of vegetation per day and "
            "drinking up to 190 litres of water. Their large ears act as radiators, helping "
            "regulate body temperature in the African heat. "
            "Elephants communicate using a range of sounds, including rumbles, roars, and "
            "infrasonic calls below the range of human hearing, which can travel for several "
            "kilometres through the ground. "
            "The African elephant is listed as vulnerable on the IUCN Red List, primarily "
            "due to habitat loss and poaching for ivory. The global population is estimated "
            "at around 415,000 individuals. In protected reserves, elephants can live up "
            "to 70 years, though wild life expectancy is often shorter."
        ),
        "qas": [
            ("How much can an adult male African elephant weigh at most?", "7,000 kilograms"),
            ("How long is the gestation period of an African elephant?", "22 months"),
            ("How much do elephant calves weigh at birth?", "120 kilograms"),
            ("How much vegetation can an African elephant consume per day?", "300 kilograms"),
            ("What is the estimated global population of African elephants?",
             "415,000 individuals"),
            ("How long can African elephants live in protected reserves?", "70 years"),
        ],
    },
    {
        "passage": (
            "The giant panda (Ailuropoda melanoleuca) is a bear species native to "
            "south-central China, found primarily in the mountainous regions of Sichuan, "
            "Shaanxi, and Gansu provinces. It is instantly recognisable by its distinctive "
            "black-and-white colouring and rounded face. Giant pandas weigh between 100 and "
            "150 kilograms and measure about 1.2 to 1.9 metres in body length. "
            "Despite being classified as carnivores, giant pandas feed almost exclusively "
            "on bamboo, consuming 12 to 38 kilograms per day. Because bamboo is low in "
            "nutrition, they spend 10 to 16 hours each day eating to meet their nutritional "
            "needs. Their digestive systems retain only about 17 percent of the energy from "
            "bamboo, compared to over 80 percent in true herbivores. "
            "Giant pandas have a very low reproductive rate. Females are fertile for only "
            "two to three days per year. Cubs are born blind and weigh only 90 to 130 grams "
            "at birth, making them one of the smallest newborns relative to mother's size "
            "among placental mammals. Mothers typically give birth to one or two cubs, but "
            "can rarely raise more than one in the wild. "
            "As of 2022, the wild giant panda population stands at approximately 1,864 "
            "individuals, all living in China. Thanks to sustained conservation efforts, "
            "the species was downgraded from 'endangered' to 'vulnerable' on the IUCN Red "
            "List in 2016. Giant pandas can live up to 20 years in the wild and around "
            "30 years in captivity."
        ),
        "qas": [
            ("How much bamboo can a giant panda consume per day?", "12 to 38 kilograms"),
            ("How many hours per day do giant pandas spend eating?", "10 to 16 hours"),
            ("How many days per year is a female giant panda fertile?", "two to three days"),
            ("What is the approximate wild giant panda population as of 2022?",
             "1,864 individuals"),
            ("In what year was the giant panda downgraded to vulnerable on the IUCN Red List?",
             "2016"),
            ("How much do giant panda cubs weigh at birth?", "90 to 130 grams"),
        ],
    },
    {
        "passage": (
            "The blue whale (Balaenoptera musculus) is the largest animal known to have "
            "ever existed on Earth. Adults can reach lengths of up to 33 metres and weigh "
            "as much as 199 metric tons, although most individuals measure between 24 and "
            "30 metres. The heart of a blue whale is roughly the size of a small car and "
            "can weigh up to 180 kilograms. "
            "Blue whales are found in all the world's major oceans, migrating seasonally "
            "between polar feeding grounds in summer and tropical breeding areas in winter. "
            "Despite their enormous size, blue whales feed primarily on tiny crustaceans "
            "called krill. A single blue whale can consume up to 40 million krill per day, "
            "equivalent to about 3,600 kilograms of food. During feeding season, they dive "
            "to depths of up to 500 metres. "
            "Blue whales produce some of the loudest sounds of any animal, with calls "
            "reaching up to 188 decibels and audible from hundreds of kilometres away. "
            "These low-frequency vocalisations are used for long-distance communication "
            "and navigation. "
            "The gestation period lasts about 10 to 12 months, and calves are born "
            "measuring approximately 7 metres in length and weighing around 2,700 kilograms. "
            "Calves gain approximately 90 kilograms per day while nursing. "
            "Hunted nearly to extinction during the twentieth century, the global blue whale "
            "population is estimated between 10,000 and 25,000 individuals. They can live "
            "for 80 to 90 years, and some individuals may reach over 100 years of age."
        ),
        "qas": [
            ("What is the maximum recorded length of a blue whale?", "33 metres"),
            ("How much krill can a blue whale eat per day in kilograms?", "3,600 kilograms"),
            ("How long does the gestation period of a blue whale last?", "10 to 12 months"),
            ("How long are blue whale calves at birth?", "7 metres"),
            ("How loud can blue whale calls be in decibels?", "188 decibels"),
            ("How long can a blue whale live?", "80 to 90 years"),
        ],
    },
    # --- SCIENTISTS & INVENTORS ----------------------------------------
    {
        "passage": (
            "Marie Curie (born Maria Sklodowska on 7 November 1867 in Warsaw, Poland) "
            "was a physicist and chemist who conducted pioneering research on radioactivity. "
            "She was the first woman to win a Nobel Prize, the only person to win Nobel "
            "Prizes in two different sciences, and the first woman to become a professor "
            "at the University of Paris. "
            "She received the Nobel Prize in Physics in 1903, shared with her husband "
            "Pierre Curie and physicist Henri Becquerel, for their research on radiation. "
            "She was awarded a second Nobel Prize, in Chemistry, in 1911 for the discovery "
            "and isolation of radium and polonium. Polonium was named after her native "
            "country Poland. "
            "Together with Pierre, she developed techniques to isolate radioactive isotopes "
            "and invented the term 'radioactivity'. Their laboratory work led to the "
            "discovery of two new elements, polonium (atomic number 84) and radium (atomic "
            "number 88), both far more radioactive than uranium. "
            "During World War I, Curie developed mobile radiography units known as 'petites "
            "Curies', providing X-ray services to wounded soldiers at the front. She "
            "personally drove and operated these units, training 150 women as radiological "
            "technicians during the war. "
            "She died on 4 July 1934 at the age of 66 from aplastic anaemia, likely caused "
            "by prolonged exposure to radiation during her research. The SI unit of "
            "radioactivity, the curie (Ci), is named in her honour."
        ),
        "qas": [
            ("In what year was Marie Curie born?", "1867"),
            ("How many Nobel Prizes did Marie Curie win?", "two"),
            ("In what year did Marie Curie win the Nobel Prize in Chemistry?", "1911"),
            ("What element did Marie Curie name after her home country?", "polonium"),
            ("What disease did Marie Curie die from?", "aplastic anaemia"),
            ("What SI unit is named in honour of Marie Curie?", "curie"),
        ],
    },
    {
        "passage": (
            "Nikola Tesla (born 10 July 1856 in Smiljan, Serbia, then part of the Austrian "
            "Empire) was an inventor and electrical engineer best known for his contributions "
            "to the design of the modern alternating current (AC) electricity supply system. "
            "Tesla showed exceptional ability in mathematics and physics from an early age. "
            "He studied at the Austrian Polytechnic in Graz and the Charles-Ferdinand "
            "University in Prague before beginning his engineering career in Europe. "
            "Tesla immigrated to the United States in 1884 and briefly worked for Thomas "
            "Edison before the two parted ways over a fundamental disagreement about AC "
            "versus direct current (DC) power. He then partnered with industrialist George "
            "Westinghouse, who purchased several of Tesla's patents and backed the "
            "commercialisation of AC electricity. "
            "Among his most significant inventions are the Tesla coil, the rotating magnetic "
            "field, and the induction motor. He held around 300 patents worldwide by the "
            "end of his career. Tesla famously won the 'War of Currents' against Edison's "
            "DC system. His AC system was demonstrated at the 1893 World's Columbian "
            "Exposition in Chicago, helping establish AC as the standard for electricity "
            "transmission worldwide. "
            "Tesla also conducted early research into radio transmission, X-rays, and "
            "wireless energy transfer. He died on 7 January 1943 in New York City at the "
            "age of 86. The SI unit of magnetic flux density, the tesla (T), is named in "
            "his honour."
        ),
        "qas": [
            ("In what year did Nikola Tesla immigrate to the United States?", "1884"),
            ("Approximately how many patents did Tesla hold by the end of his career?", "300"),
            ("In what year was Tesla's AC system demonstrated at the World's Columbian Exposition?",
             "1893"),
            ("In what city did Nikola Tesla die?", "New York City"),
            ("At what age did Nikola Tesla die?", "86"),
            ("What SI unit is named after Nikola Tesla?", "tesla"),
        ],
    },
    {
        "passage": (
            "Albert Einstein (born 14 March 1879 in Ulm, in the Kingdom of Wuerttemberg in "
            "the German Empire) was a theoretical physicist who developed the theory of "
            "relativity, one of the two pillars of modern physics alongside quantum mechanics. "
            "Einstein showed an early aptitude for mathematics and physics, teaching himself "
            "calculus by age 15. He completed his doctorate at the University of Zurich in "
            "1905, the same year he published four groundbreaking papers, now known as his "
            "annus mirabilis papers. "
            "Einstein is best known for his mass-energy equivalence formula E = mc², "
            "described as 'the world's most famous equation'. The formula emerged from his "
            "special theory of relativity and expresses the relationship between mass and "
            "energy. He later developed the general theory of relativity, published in 1915, "
            "which described gravity as a curvature of spacetime. "
            "He received the Nobel Prize in Physics in 1921, not for relativity but for his "
            "discovery of the law of the photoelectric effect, which was crucial to the "
            "development of quantum theory. "
            "After Hitler's rise to power, Einstein emigrated from Germany to the United "
            "States in 1933 and accepted a position at the Institute for Advanced Study in "
            "Princeton, New Jersey. He became an American citizen in 1940. "
            "Einstein died on 18 April 1955 in Princeton at the age of 76. Time magazine "
            "named him the Person of the Century in 1999."
        ),
        "qas": [
            ("In what year was Albert Einstein born?", "1879"),
            ("For what discovery did Einstein receive the 1921 Nobel Prize in Physics?",
             "the photoelectric effect"),
            ("In what year did Einstein emigrate from Germany to the United States?", "1933"),
            ("In what year did Einstein become an American citizen?", "1940"),
            ("At what age did Albert Einstein die?", "76"),
            ("What did Time magazine name Einstein in 1999?", "Person of the Century"),
        ],
    },
    # --- GEOGRAPHY & COUNTRIES -----------------------------------------
    {
        "passage": (
            "Japan is an archipelago located in East Asia, consisting of 6,852 islands. "
            "The four main islands are Honshu, Hokkaido, Kyushu, and Shikoku, which together "
            "account for 97 percent of the country's total land area. The total land area "
            "of Japan is approximately 377,975 square kilometres. "
            "Japan has a population of approximately 125 million people, making it the "
            "eleventh most populous country in the world. The capital city is Tokyo, "
            "which is also the most populous metropolitan area on Earth with a population "
            "of around 37 million people. Japanese society is notable for its very high "
            "life expectancy, with the average reaching 84 years as of the early 2020s. "
            "Japan has the world's third-largest economy by nominal GDP. Major industries "
            "include automotive manufacturing, electronics, and robotics. The country is "
            "home to some of the world's largest corporations, including Toyota, Sony, "
            "Honda, and Mitsubishi. "
            "Japan is located in the Pacific Ring of Fire, making it one of the world's "
            "most seismically active countries, experiencing around 1,500 earthquakes per "
            "year. Mount Fuji, the country's highest peak and an active stratovolcano, "
            "stands at 3,776 metres above sea level. "
            "Japan has 23 UNESCO World Heritage Sites. The country uses the Japanese yen "
            "(JPY) as its currency and became a member of the United Nations in 1956."
        ),
        "qas": [
            ("How many islands make up Japan?", "6,852 islands"),
            ("What is the total land area of Japan in square kilometres?",
             "377,975 square kilometres"),
            ("What is the approximate population of Japan?", "125 million"),
            ("How tall is Mount Fuji?", "3,776 metres"),
            ("How many UNESCO World Heritage Sites does Japan have?", "23"),
            ("In what year did Japan become a member of the United Nations?", "1956"),
        ],
    },
    {
        "passage": (
            "Brazil is the largest country in South America and the fifth-largest in the "
            "world by both total area and population. It covers approximately 8.5 million "
            "square kilometres, spanning four time zones. Brazil borders every South American "
            "country except Chile and Ecuador, giving it a vast network of land borders "
            "totalling around 16,885 kilometres. "
            "Brazil has a population of around 215 million people, making it the sixth most "
            "populous country globally. The official language is Portuguese, inherited from "
            "the period of Portuguese colonisation that began in 1500. The capital city is "
            "Brasilia, a planned city inaugurated on 21 April 1960, when the capital was "
            "moved from Rio de Janeiro. "
            "The Amazon River, which flows through northern Brazil, is the world's largest "
            "river by discharge volume, releasing around 20 percent of all freshwater "
            "entering the oceans. The Amazon rainforest covers approximately 5.5 million "
            "square kilometres and is home to an estimated 10 percent of all species on Earth. "
            "Brazil's economy is the largest in Latin America and the twelfth largest in "
            "the world. Key industries include agriculture, mining, petroleum, and "
            "manufacturing. Brazil is the world's largest producer of coffee, oranges, and "
            "sugarcane. "
            "Brazil has won the FIFA World Cup a record five times, in 1958, 1962, 1970, "
            "1994, and 2002. The country hosted the Summer Olympic Games in 2016 in Rio "
            "de Janeiro."
        ),
        "qas": [
            ("What is the approximate area of Brazil in square kilometres?",
             "8.5 million square kilometres"),
            ("What is the population of Brazil?", "215 million"),
            ("When was Brasilia inaugurated?", "21 April 1960"),
            ("How many times has Brazil won the FIFA World Cup?", "five times"),
            ("In what year did Brazil host the Summer Olympic Games?", "2016"),
            ("How large is the Amazon rainforest in square kilometres?",
             "5.5 million square kilometres"),
        ],
    },
    {
        "passage": (
            "Canada is the second-largest country in the world by total area, covering "
            "approximately 9.98 million square kilometres. It is divided into ten provinces "
            "and three territories, stretching from the Atlantic Ocean in the east to the "
            "Pacific Ocean in the west, and northward into the Arctic Ocean. "
            "Canada has a population of around 38 million people. The capital city is Ottawa, "
            "located in the province of Ontario. The country has two official languages: "
            "English and French. French is spoken predominantly in the province of Quebec "
            "and parts of New Brunswick. "
            "Canada has the world's longest coastline, stretching approximately 202,080 "
            "kilometres, including the mainland coast and the coasts of offshore islands. "
            "It also shares the world's longest international border with the United States, "
            "running approximately 8,891 kilometres. "
            "The CN Tower in Toronto, standing 553 metres tall, was the world's tallest "
            "free-standing structure for 32 years from its completion in 1976 until it was "
            "surpassed in 2008. "
            "Canada is a member of the G7, NATO, the Commonwealth of Nations, and the "
            "United Nations. The country's economy is the ninth-largest in the world. "
            "Major exports include petroleum, natural gas, automobiles, and agricultural "
            "products, particularly wheat and canola. "
            "Canada has 20 UNESCO World Heritage Sites. The Canadian dollar (CAD) is the "
            "official currency."
        ),
        "qas": [
            ("What is the total area of Canada in square kilometres?",
             "9.98 million square kilometres"),
            ("How many provinces does Canada have?", "ten provinces"),
            ("What are Canada's two official languages?", "English and French"),
            ("How long is Canada's coastline?", "202,080 kilometres"),
            ("For how many years was the CN Tower the world's tallest free-standing structure?",
             "32 years"),
            ("How many UNESCO World Heritage Sites does Canada have?", "20"),
        ],
    },
    # --- INVENTIONS & TECHNOLOGY ---------------------------------------
    {
        "passage": (
            "The World Wide Web was invented by British scientist Sir Tim Berners-Lee in "
            "1989 while working at CERN, the European particle physics laboratory near "
            "Geneva, Switzerland. Berners-Lee proposed the system in March 1989 as a way "
            "to help scientists share information more efficiently across the institution's "
            "distributed computing infrastructure. "
            "The first web browser, called WorldWideWeb (later renamed Nexus), was created "
            "by Berners-Lee himself in 1990 on a NeXT computer. The first website, "
            "info.cern.ch, went online on 20 December 1990. The site described the "
            "WorldWideWeb project itself and explained how to create web pages. "
            "Crucially, Berners-Lee made the web technology freely available, refusing to "
            "patent it. This decision was endorsed by CERN and proved decisive in enabling "
            "the rapid global adoption of the web as an open standard. "
            "By 1993, the number of websites had grown to 130. The release of the Mosaic "
            "browser that year, the first to display images inline with text, triggered "
            "explosive public interest. The HTTP protocol, which the web relies on, was "
            "first specified in 1991. "
            "Berners-Lee was awarded a knighthood by Queen Elizabeth II in 2004. He "
            "received the Turing Award, often described as the 'Nobel Prize of computing', "
            "in 2016. "
            "As of 2023, there are approximately 1.9 billion websites on the internet, "
            "though only a fraction are actively maintained."
        ),
        "qas": [
            ("In what year did Tim Berners-Lee invent the World Wide Web?", "1989"),
            ("At which institution did Tim Berners-Lee invent the web?", "CERN"),
            ("What was the first website ever created?", "info.cern.ch"),
            ("When did the first website go online?", "20 December 1990"),
            ("How many websites existed by 1993?", "130"),
            ("In what year did Tim Berners-Lee receive the Turing Award?", "2016"),
        ],
    },
    {
        "passage": (
            "The International Space Station (ISS) is a modular space station in low Earth "
            "orbit, serving as a microgravity and space environment research laboratory. "
            "Construction began in November 1998 with the launch of the Russian Zarya "
            "control module, which provided initial propulsion and power. The connecting "
            "Unity module, contributed by NASA, followed in December 1998. "
            "The station has been continuously inhabited since 2 November 2000, making it "
            "the longest continuous human presence in space. Early crews arrived via the "
            "Russian Soyuz spacecraft, which continues to serve as the primary crew vehicle. "
            "The ISS orbits Earth at an altitude of approximately 408 kilometres and travels "
            "at a speed of about 7.66 kilometres per second, completing about 15.5 orbits "
            "per day. Astronauts on board experience approximately 16 sunrises and sunsets "
            "every 24 hours. "
            "The station has a pressurised volume of 916 cubic metres — roughly equivalent "
            "to a six-bedroom house — and a mass of approximately 420,000 kilograms. Its "
            "solar arrays span 109 metres, wider than an American football field. "
            "The ISS has been visited by more than 270 individuals from 20 different "
            "countries. It is a joint project involving five space agencies: NASA (USA), "
            "Roscosmos (Russia), ESA (Europe), JAXA (Japan), and CSA (Canada). "
            "The total cost of the ISS project is estimated at over 150 billion US dollars."
        ),
        "qas": [
            ("In what year did construction of the ISS begin?", "1998"),
            ("Since what date has the ISS been continuously inhabited?", "2 November 2000"),
            ("At what altitude does the ISS orbit Earth?", "408 kilometres"),
            ("What is the mass of the ISS in kilograms?", "420,000 kilograms"),
            ("How wide are the ISS solar arrays?", "109 metres"),
            ("How many individuals have visited the ISS?", "more than 270"),
        ],
    },
    # --- HISTORY -------------------------------------------------------
    {
        "passage": (
            "The Great Wall of China is a series of fortifications built along the northern "
            "borders of China to protect against nomadic invasions from the Eurasian Steppe. "
            "Construction of various walls began as early as the 7th century BC, during the "
            "Spring and Autumn Period, when regional states built their own defences. "
            "The first unified wall was ordered by Emperor Qin Shi Huang around 221 BC, "
            "connecting and extending earlier sections. However, the most well-known and "
            "best-preserved sections were built during the Ming dynasty (1368-1644), using "
            "brick and stone rather than earlier earthen construction. "
            "The total length of all sections of the Great Wall, including all branches "
            "and parallel walls, is approximately 21,196 kilometres. The wall averages "
            "6 to 7 metres in height and 4 to 5 metres in width. Watchtowers were built "
            "at regular intervals for communication and defence. "
            "It has been estimated that up to 400,000 workers died during the construction "
            "of the Ming-era sections alone, many of whom were buried within or near the "
            "wall itself. Workers included soldiers, peasants, and prisoners. "
            "UNESCO inscribed the Great Wall on the World Heritage List in 1987. It was "
            "declared one of the New Seven Wonders of the World in a global poll in 2007. "
            "The Great Wall is the world's longest man-made structure and is visible from "
            "low Earth orbit under favourable conditions."
        ),
        "qas": [
            ("When did construction of the Great Wall begin?", "7th century BC"),
            ("During which dynasty were the most well-known sections built?", "Ming dynasty"),
            ("What is the total length of all Great Wall sections?", "21,196 kilometres"),
            ("How tall does the Great Wall average?", "6 to 7 metres"),
            ("In what year did UNESCO inscribe the Great Wall on the World Heritage List?", "1987"),
            ("In what year was the Great Wall declared one of the New Seven Wonders of the World?",
             "2007"),
        ],
    },
    {
        "passage": (
            "The Roman Colosseum, officially known as the Flavian Amphitheatre, is an oval "
            "amphitheatre in the centre of Rome, Italy. It is the largest amphitheatre ever "
            "built and is considered one of the greatest works of Roman engineering and "
            "architecture, representing a significant achievement in construction technique "
            "and crowd management. "
            "Construction began under Emperor Vespasian around 70 AD and was completed in "
            "80 AD under his son and successor Titus, who inaugurated the amphitheatre with "
            "100 days of games. A further expansion was added by Emperor Domitian between "
            "81 and 96 AD, adding an additional tier. "
            "The Colosseum could hold between 50,000 and 80,000 spectators, with an average "
            "audience of around 65,000. It measures 188 metres long, 156 metres wide, and "
            "48 metres tall. The outer wall required over 100,000 cubic metres of travertine "
            "stone. An elaborate system of underground passages, called the hypogeum, housed "
            "animals, gladiators, and stage machinery. "
            "The Colosseum was used for gladiatorial contests, animal hunts called venationes, "
            "public executions, and dramatic performances. It is estimated that over 400,000 "
            "people and more than one million animals died within the Colosseum during the "
            "roughly 400 years of its active use. "
            "Today the Colosseum is Rome's most popular tourist attraction, receiving "
            "approximately 7 million visitors per year. It has been listed as a UNESCO "
            "World Heritage Site since 1980."
        ),
        "qas": [
            ("In what year did construction of the Colosseum begin?", "70 AD"),
            ("In what year was the Colosseum completed?", "80 AD"),
            ("How many spectators could the Colosseum hold at maximum?", "80,000 spectators"),
            ("How tall is the Colosseum?", "48 metres"),
            ("How many visitors does the Colosseum receive per year?", "7 million visitors"),
            ("How many people are estimated to have died within the Colosseum?",
             "over 400,000 people"),
        ],
    },
    # --- SCIENCE & NATURE --------------------------------------------
    {
        "passage": (
            "The Mariana Trench is the deepest oceanic trench on Earth, located in the "
            "western Pacific Ocean, east of the Mariana Islands. Its deepest known point "
            "is the Challenger Deep, which reaches a depth of approximately 11,034 metres "
            "below sea level. This is deeper than Mount Everest is tall. "
            "The trench is about 2,550 kilometres long and 69 kilometres wide on average, "
            "making it one of the largest geological features on Earth's surface. It was "
            "formed by the subduction of the Pacific tectonic plate beneath the Mariana "
            "plate. "
            "The Mariana Trench was first surveyed in 1875 by the HMS Challenger expedition, "
            "using weighted rope to measure depth. In 1960, oceanographer Jacques Piccard "
            "and US Navy Lieutenant Don Walsh became the first humans to descend to the "
            "Challenger Deep in the bathyscaphe Trieste. The descent took approximately "
            "4 hours and 47 minutes to reach the bottom. "
            "Water pressure at the bottom of the Challenger Deep is about 1,086 bar, "
            "equivalent to more than 1,000 times the standard atmospheric pressure at sea "
            "level. Despite these extreme conditions, microorganisms and small organisms "
            "such as amphipods have been found living in the trench. Over 200 microorganism "
            "species have been discovered there. "
            "In 2012, film director James Cameron made a solo dive to the Challenger Deep "
            "in the purpose-built submersible Deepsea Challenger, becoming the third person "
            "to reach the bottom and the first to do so alone."
        ),
        "qas": [
            ("What is the maximum depth of the Mariana Trench?", "11,034 metres"),
            ("How long is the Mariana Trench?", "2,550 kilometres"),
            ("When was the Mariana Trench first surveyed?", "1875"),
            ("In what year did Jacques Piccard and Don Walsh first descend to the Challenger Deep?",
             "1960"),
            ("How long did the 1960 dive take to reach the bottom?", "4 hours and 47 minutes"),
            ("In what year did James Cameron dive to the Challenger Deep?", "2012"),
        ],
    },
    {
        "passage": (
            "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
            "rainforest covering most of the Amazon basin of South America. This basin "
            "encompasses 7 million square kilometres, of which 5.5 million square kilometres "
            "are covered by rainforest, making it the world's largest tropical rainforest. "
            "The Amazon represents more than half of the planet's remaining rainforests and "
            "is sometimes described as the 'lungs of the Earth', though scientists note that "
            "it consumes as much oxygen as it produces. "
            "The forest is home to an estimated 10 percent of the world's known species, "
            "including around 40,000 plant species, 1,300 bird species, 3,000 types of "
            "fish, and more than 430 mammals. Scientists believe many species remain "
            "undiscovered in its interior. "
            "The Amazon River, which runs through the forest, discharges about 20 percent "
            "of all fresh water entering the world's oceans. Its basin is drained by more "
            "than 1,100 tributaries. "
            "The rainforest plays a critical role in regulating global climate by absorbing "
            "carbon dioxide from the atmosphere. However, deforestation driven by agriculture, "
            "logging, and mining has reduced its extent significantly. Brazil alone lost "
            "approximately 11,568 square kilometres of rainforest in 2022. "
            "Nine nations share the Amazon basin: Brazil, Peru, Colombia, Venezuela, Ecuador, "
            "Bolivia, Guyana, Suriname, and French Guiana. Brazil contains about 60 percent "
            "of the total rainforest area."
        ),
        "qas": [
            ("How many square kilometres does the Amazon rainforest cover?",
             "5.5 million square kilometres"),
            ("Approximately what percentage of the world's known species live in the Amazon?",
             "10 percent"),
            ("How many bird species are found in the Amazon?", "1,300 bird species"),
            ("What percentage of fresh water entering the world's oceans "
             "comes from the Amazon River?",
             "20 percent"),
            ("How many square kilometres of rainforest did Brazil lose in 2022?",
             "11,568 square kilometres"),
            ("What percentage of the Amazon rainforest is contained in Brazil?", "60 percent"),
        ],
    },
    # --- EVENTS & INSTITUTIONS ----------------------------------------
    {
        "passage": (
            "The Olympic Games are the world's foremost international multi-sport event, "
            "held every four years and alternating between the Summer and Winter Games on "
            "a two-year cycle. The modern Olympics were revived by French educator Pierre de "
            "Coubertin, who was inspired by the ancient Greek games held at Olympia from "
            "776 BC until their abolition in 393 AD. "
            "The first modern Olympic Games were held in Athens, Greece, in April 1896. "
            "Only 241 athletes from 14 nations participated in those inaugural games, "
            "competing in 43 events across nine sports. No women competed, as the games "
            "were restricted to male athletes at the time. "
            "The 2020 Summer Olympics, held in Tokyo in 2021 due to the COVID-19 pandemic, "
            "featured 11,656 athletes from 206 countries and territories competing in "
            "339 events across 33 sports. Women accounted for nearly 49 percent of the "
            "participants, reflecting the dramatic change in gender inclusivity since 1896. "
            "The United States has won the most total Olympic medals in history, with a "
            "total exceeding 2,600 medals across Summer and Winter Games combined. "
            "The Olympic motto is 'Citius, Altius, Fortius', which is Latin for 'Faster, "
            "Higher, Stronger'. A fourth word, Communiter (Together), was added to the "
            "motto in 2021. The five interlocking rings represent the five continents of "
            "the world united by the Olympic movement."
        ),
        "qas": [
            ("In what year were the first modern Olympic Games held?", "1896"),
            ("In which city were the first modern Olympics held?", "Athens, Greece"),
            ("How many athletes participated in the 1896 Olympics?", "241 athletes"),
            ("How many athletes competed in the 2020 Tokyo Olympics?", "11,656 athletes"),
            ("What does the Olympic motto Citius, Altius, Fortius mean in English?",
             "Faster, Higher, Stronger"),
            ("In what year was Communiter added to the Olympic motto?", "2021"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(seed: int = 42) -> list[dict[str, Any]]:
    """Generate all unique knowledge relay samples.

    Returns every (passage, question, answer) triple from _CORPUS exactly
    once — no cycling, no repetition.  With 16 passages × 6 QA pairs the
    total is 96 unique samples.

    Splitting MUST be done at the passage level via passage_id.  See the
    module docstring and m1_data_generation.split_knowledge_relay().

    Args:
        seed: Random seed used to shuffle QA order within each passage.

    Returns:
        List of sample dicts with keys:
            passage, question, answer, passage_id, relay_point.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    for idx, item in enumerate(_CORPUS):
        passage: str = item["passage"]
        qas = list(item["qas"])
        rng.shuffle(qas)  # shuffle QA order within passage for variety
        for question, answer in qas:
            samples.append({
                "passage":    passage,
                "question":   question,
                "answer":     answer,
                "passage_id": idx,
                "relay_point": RELAY_POINT,
            })

    return samples
