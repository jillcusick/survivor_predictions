# ingest cleaned data frame and create transformed features for modeling 

import pandas as pd

# define mapping dictionary 
FULL_NAME_REGION_MAPPING = {
    # Northeast
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast', 
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast', 
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    
    # Midwest
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest', 
    'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest', 
    'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest', 
    'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    
    # South
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South', 'Maryland': 'South', 
    'North Carolina': 'South', 'South Carolina': 'South', 'Virginia': 'South', 
    'District of Columbia': 'South', 'D.C.': 'South', 'West Virginia': 'South', 'Alabama': 'South', 
    'Kentucky': 'South', 'Mississippi': 'South', 'Tennessee': 'South', 
    'Arkansas': 'South', 'Louisiana': 'South', 'Oklahoma': 'South', 'Texas': 'South',
    
    # West
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West', 'Montana': 'West', 
    'Nevada': 'West', 'New Mexico': 'West', 'Utah': 'West', 'Wyoming': 'West', 
    'Alaska': 'West', 'California': 'West', 'Hawaii': 'West', 'Oregon': 'West', 'Washington': 'West'
}

def get_region_from_state(state):
    """Helper function to map state to a region."""
    if pd.isna(state):
        return 'Unknown/International'
    return FULL_NAME_REGION_MAPPING.get(state, 'Unknown/International')

def get_industry_from_occupation(occupation):
    """Helper function to map occupation to an industry cluster."""
    if pd.isna(occupation):
        return "Other"
    
    occupation = str(occupation).lower()
    
    # Physical (athletes, tactical, protective services, fitness, labor/trades)
    if any(k in occupation for k in [
        'athlete', 'trainer', 'fire', 'police', 'cop', 'coach', 'fitness', 'body',
        'military', 'army', 'navy', 'marine', 'air force', 'guard', 'trooper', 
        'security', 'surfer', 'martial', 'fighter', 'olympian', 'pro ', 'stunt',
        'construction', 'mechanic', 'carpenter', 'plumber', 'electrician', 
        'contractor', 'farmer', 'fisherman', 'driver', 'pilot', 'laborer', 
        'welder', 'painter', 'landscape', 'worker', 'deckhand', 'inspector',
        'fisherman', 'veteran', 'paratrooper', 'sheriff', 'yogi', 'nfl', 'player', 'nba',
        'wellness', 'volleyball', 'racer', 'captain', 'paralympian', 'medalist', 'maintenance'
    ]):
        return "Physical"
        
    # Corporate (business, sales, finance, management, legal, real estate)
    elif any(k in occupation for k in [
        'exec', 'manager', 'sales', 'real estate', 'lawyer', 'consultant', 
        'finance', 'ceo', 'director', 'business', 'entrepreneur', 'banker', 
        'analyst', 'marketing', 'recruiter', 'accountant', 'insurance', 'broker', 
        'agent', 'executive', 'corporate', 'trader', 'investor', 'attorney',
        'legal', 'project', 'operations', 'strategist', 'fundraiser', 'judge',
        'politician'
    ]):
        return "Corporate"
        
    # Creative (media, writing, arts, entertainment, design)
    elif any(k in occupation for k in [
        'writer', 'actor', 'actress', 'model', 'artist', 'media', 'journalist', 'musician',
        'singer', 'dj', 'producer', 'host', 'dancer', 'blogger', 'influencer', 
        'photographer', 'author', 'comedian', 'designer', 'fashion', 'radio',
        'filmmaker', 'editor', 'podcaster', 'content', 'performer', 'magician',
        'reporter', 'youtube', 'commentator', 'star', 'gamer', 'videographer',
        'showgirl', 'miss', 'anchor', 'speaker'
    ]):
        return "Creative"
        
    # Academic (students, teachers, professors, researchers, education)
    elif any(k in occupation for k in [
        'student', 'teacher', 'professor', 'research', 'researcher', 'educator', 
        'principal', 'dean', 'academic', 'tutor', 'instructor', 'scholar',
        'kindergarten', 'coach', 'school', 'university', 'college', 'dean',
        'graduate', 'phd'
    ]):
        return "Academic"
        
    # Medical/STEM (healthcare, engineering, science, IT, technology)
    elif any(k in occupation for k in [
        'nurse', 'doctor', 'engineer', 'scientist', 'medical', 'dentist', 
        'surgeon', 'pharmacist', 'therapist', 'chemist', 'biologist', 'physics', 
        'vet', 'software', 'programmer', 'it ', 'tech', 'technician', 'analyst',
        'biochemist', 'pathologist', 'psychiatrist', 'psychologist', 'hygienist',
        'radiologic', 'paramedic', 'emt', 'medic', 'surgeon', 'developer', 'dietician',
        'dietitian', 'astronaut', 'physician'
    ]) or 'ologist' in occupation or 'olomist' in occupation:
        return "Medical/STEM"
        
    # Service/Hospitality/Relationships (food service, beauty, counseling, support, admin)
    elif any(k in occupation for k in [
        'bartender', 'server', 'flight attendant', 'host', 'service', 'waiter', 
        'hair', 'stylist', 'counselor', 'secretary', 'waitress', 'barista', 
        'hospitality', 'hotel', 'restaurant', 'caterer', 'nanny', 'social worker', 
        'minister', 'pastor', 'clergy', 'therapist', 'assistant', 'receptionist',
        'casino', 'dealer', 'prowler', 'flight', 'stewardess', 'concierge',
        'orphan', 'advocate', 'volunteer', 'community', 'nonprofit', 'charity',
        'coordinator', 'administrative', 'clerk', 'office', 'human resources', 'hr', 'owner', 
        'barber', 'cosmetologist', 'caretaker', 'guide', 'chef', 'promoter', 'vendor', 'planner',
        'mentor', 'homemaker', 'home', 'staff', 'mom'
    ]):
        return "Service/Hospitality"
        
    else:
        return "Other"


def engineer_features(df): 
    ''' Ingest cleaned data frame and create transformed features and outcomes for modeling '''

    ### standardize target outcome variable for regression models 
    
    # calculate season cast size 
    df['cast_size'] = df.groupby('season')['castaway_id'].transform('count')
    
    # convert raw placement outcome to placement score relative to number of players on season 
    df["place_score"] = (df["cast_size"] - df["place"] + 1) / df["cast_size"]
    
    ### create target outcome variable for classification models
    
    # create binary outcome variable for whether castaway made season "merge" (i.e. jury) or not
    df['made_merge'] = ((df['jury'] == 1) | (df['finalist'] == 1)).astype(int)
    
    ### engineer predictors 
    
    # relative age: calculate players' age relative to the mean age of their season
    df['rel_age'] = df['age'] - df.groupby('season')['age'].transform('mean')
    
    # geographical region: map state to region of US
    df['region'] = df['state'].apply(get_region_from_state)
    
    # industry: map occupation to industry cluster
    df['industry'] = df['occupation'].apply(get_industry_from_occupation)
        
    # gender: update gender for 2 non-binary players (played the game as female) to allow for binary variable encoding
    df['gender'] = df['gender'].replace('Non-binary', 'Female')
    
    # season "era": create binary feature for whether season was "new" or "old" era
    df['new_era'] = (df['season'] >= 41).astype(int) 
    
    
    
    return df
    