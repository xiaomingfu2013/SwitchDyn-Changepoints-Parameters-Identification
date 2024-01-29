using HTTP, Dates
using DataFrames, CSV
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data-old.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# select Germany data, keep only the columns we need
# ["new_cases", "new_deaths", "weekly_hosp_admissions"]
df_germany = df[df.location .== "Germany", [
        :date,
        :new_cases,
        :new_deaths,
        :weekly_hosp_admissions
    ]
]
# select time from 2020-03-01 to 2020-06-30
condition = (df_germany.date .>= Date("2020-03-01")) .& (df_germany.date .< Date("2020-08-01"))

df_germany = df_germany[condition, :]
save_path = datadir("raw_data/germany_data.csv")

CSV.write(save_path, df_germany)
