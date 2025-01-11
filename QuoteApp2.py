import streamlit as st

# Function to get user inputs
def get_user_inputs():
    job_type = st.selectbox("Select Job Type", ["Lawn Care", "Tree Trimming", "Garden Design"])
    size_category = st.selectbox("Select Size Category", ["Small", "Medium", "Large"])
    terrain_complexity = st.selectbox("Select Terrain Complexity", ["Flat", "Sloped", "Rocky"])
    special_requests = st.text_area("Special Requests")
    return job_type, size_category, terrain_complexity, special_requests

# Function to calculate the quote
def calculate_quote(job_type, size_category, terrain_complexity, special_requests):
    # Base costs for each job type
    base_costs = {
        "Lawn Care": 50,
        "Tree Trimming": 100,
        "Garden Design": 150
    }
    # Multipliers based on size category
    size_multipliers = {
        "Small": 1.0,
        "Medium": 1.5,
        "Large": 2.0
    }
    # Multipliers based on terrain complexity
    terrain_multipliers = {
        "Flat": 1.0,
        "Sloped": 1.2,
        "Rocky": 1.5
    }
    # Fetch base cost, size multiplier, and terrain multiplier
    base_cost = base_costs.get(job_type, 0)
    size_multiplier = size_multipliers.get(size_category, 1)
    terrain_multiplier = terrain_multipliers.get(terrain_complexity, 1)
    # Calculate the total quote
    quote = base_cost * size_multiplier * terrain_multiplier
    # Adjust quote based on special requests (e.g., adding 10% for special requests)
    if special_requests:
        quote *= 1.1
    return quote

# Function to display the quote
def display_quote(quote):
    st.write(f"**Estimated Quote: ${quote:.2f}**")

# Function to track performance metrics
import time
def track_performance():
    start_time = time.time()
    # Simulate quote calculation
    time.sleep(0.1)
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"**Execution Time: {execution_time:.4f} seconds**")

# Main function to run the application
def main():
    st.title("Landscaping Quote Application")
    job_type, size_category, terrain_complexity, special_requests = get_user_inputs()
    quote = calculate_quote(job_type, size_category, terrain_complexity, special_requests)
    display_quote(quote)
    track_performance()

if __name__ == "__main__":
    main()

