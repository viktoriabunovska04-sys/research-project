BULGARIA = """
Anyone who, by speech, press or other media, by electronic information systems or in another manner, propagates or incites discrimination, violence or hatred on the grounds of race, nationality or ethnic origin shall be punishable by imprisonment of one to four years and a fine from BGN 5,000 to 10,000, as well as public censure.
A person who propagates or instigates discrimination, violence or hatred on religious basis by speech, through the press or other mass media, through electronic information systems or in another way, shall be punished by imprisonment for up to four years or probation and a fine from BGN five thousand to ten thousand.
"""

CROATIA = """
Whoever in print, through radio, television, computer system or network, at a public gathering or in some other way publicly incites to or makes available to the public tracts, pictures or other material instigating violence or hatred directed against a group of persons or a member of such a group on account of their race, religion, national or ethnic origin, descent, colour, gender, sexual orientation, gender identity, disability or any other characteristics shall be punished by imprisonment not exceeding three years.
    (2) The same punishment as referred to in paragraph 1 of this Article shall be inflicted on whoever publicly approves of, denies or grossly trivializes the crimes of genocide, crimes of aggression, crimes against humanity or war crimes, directed against a group of persons or a member of such a group on account of their race, religion, national or ethnic origin, descent or colour in a manner likely to incite to violence or hatred against such a group or a member of such a group.
    (3) The attempt to carry out or commit criminal offences referred to in paragraph 1 and 2 of this Article shall be punishable
"""

META = """
We define hateful conduct as direct attacks against people — rather than concepts or institutions — on the basis of what we call protected characteristics (PCs): race, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and serious disease. Additionally, we consider age a protected characteristic when referenced along with another protected characteristic. We also protect refugees, migrants, immigrants, and asylum seekers from the most severe attacks (Tier 1 below), though we do allow commentary on and criticism of immigration policies. Similarly, we provide some protections for non- protected characteristics, such as occupation, when they are referenced along with a protected characteristic. Sometimes, based on local nuance, we consider certain words or phrases as frequently used proxies for protected characteristics.
We remove dehumanizing speech, allegations of serious immorality or criminality, and slurs. We also remove harmful stereotypes, which we define as dehumanizing comparisons that have historically been used to attack, intimidate, or exclude specific groups, and that are often linked with offline violence. Finally, we remove serious insults, expressions of contempt or disgust, cursing, and calls for exclusion or segregation when targeting people based on protected characteristics. We separate this speech into two tiers of severity, described below.
We recognize that people sometimes share content that includes slurs or someone else’s speech in order to condemn the speech or report on it. In other cases, speech, including slurs, that might otherwise violate our standards is used self-referentially or in an empowering way. We allow this type of speech where the speaker’s intention is clear. Where intention is unclear, we may remove content.
"""

REDDIT = """
Remember the human. Reddit is a place for creating community and belonging, not for attacking marginalized or vulnerable groups of people. Everyone has a right to use Reddit free of harassment, bullying, and threats of violence. Communities and people that incite violence or that promote hate based on identity or vulnerability will be banned.
Marginalized or vulnerable groups include, but are not limited to, groups based on their actual and perceived race, color, religion, national origin, ethnicity, immigration status, gender, gender identity, sexual orientation, pregnancy, or disability. These include victims of a major violent event and their families. 
While the rule on hate protects such groups, it does not protect those who promote attacks of hate or who try to hide their hate in bad faith claims of discrimination.
Some examples of hateful activities that would violate the rule:
•	Community dedicated to mocking people with physical disabilities. 
•	Post describing a racial minority as sub-human and inferior to the racial majority.
•	Comment arguing that rape of women should be acceptable and not a crime.
•	Meme declaring that it is sickening that people of color have the right to vote.
•	Post promoting harmful tropes or generalizations based on religion (e.g. a certain religious group controls the media, or consists entirely of terrorists). 
•	A comment denying or minimizing the scale of a hate-based violent event. 
Additionally, when evaluating the activity of a community or an individual user, we consider both the context as well as the pattern of behavior. 
"""

THEORETIC_INCLUSION = """
Hate speech is defined as language targeted at any person or group based on their race, colour, religion, nationality, ethnicity, sexual orientation, and disability, and incites violence and hatred or insults a group through the usage of group characteristics and slurs. If a slur or group characteristic is used by a member of the targeted group it is not considered hate speech; however if the speaker's group membership is unknown this exception does not apply.
"""

THEORETIC_INCLUSION_EXCLUSION = """
Hate speech is defined as language targeted at a dominant or stigmatized person or group based on their race, colour, religion, nationality, ethnicity, sexual orientation, and disability, but not gender, language, or class. It incites violence and hatred but not discrimination, or insults a group through the usage of group characteristics and slurs but not stereotypes. If a slur or group characteristic is used by a member of the targeted group it is not considered hate speech; however if the speaker's group membership is unknown this exception does not apply.
"""

DEFINITIONS = {
    "Bulgaria": BULGARIA,
    "Croatia": CROATIA,
    "Meta": META,
    "Reddit": REDDIT,
    "Theoretic Inclusion": THEORETIC_INCLUSION,
    "Theoretic Inclusion + Exclusion": THEORETIC_INCLUSION_EXCLUSION,
}