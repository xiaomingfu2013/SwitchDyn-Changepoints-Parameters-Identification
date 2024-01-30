using Plots
using DrWatson
using DataFrames
using CSV
@quickactivate "SwitchDyn-Changepoints-Parameters-Identification"
function get_germany_timeseries_data_daily(;
    viz=false,
    per_million=false,
    saveflag=false,
    color_palette=["#1f78b4" "#ff7f00" "#e31a1c"],
    percentage_missing=0.95,
    country_population = 83369840
    # "https://population.un.org/wpp/Download/Standard/CSV/"
)

    country_data = CSV.read(datadir("raw_data/germany_data.csv"), DataFrame)
    for col in names(country_data)[2:end]
        if eltype(country_data[:, col]) == Missing
            continue
        elseif eltype(country_data[:, col]) == Union{Missing,Float64}
            if count(ismissing, country_data[:, col]) >
                size(country_data, 1) * percentage_missing
                country_data[!, col] = fill(missing, size(country_data, 1))
                @info "Column $col is missing more than $(100*percentage_missing)% of the data, return as missing"
                continue
            end
        end
        country_data[:, col] = convert(
            Vector{Float64}, coalesce.(country_data[:, col], 0.0)
        )
        disallowmissing!(country_data, col)
        if per_million
            country_data[:, col] = country_data[:, col] ./ country_population * 1e6
        end
    end

    df_cum_daily = DataFrame()
    df_cum_daily[!, :date] = country_data.date
    df_cum_daily[!, :cum_daily_cases] = cumsum(country_data.new_cases)
    df_cum_daily[!, :cum_daily_deaths] = cumsum(country_data.new_deaths)
    df_cum_daily[!, :cum_daily_hospital_admissions] = cumsum(
        country_data.weekly_hosp_admissions / 7
    ) # !! weekly_hosp_admissions is weekly data, need to divide by 7 to get daily data

    if viz
        theme(:vibrant)
        plt1 = Plots.plot(
            country_data.date,
            [
                country_data.new_cases,
                country_data.weekly_hosp_admissions / 7,
                country_data.new_deaths,
            ];
            label=["daily cases" "hospital_daily_admissions" "daily death"],
            title="Daily new cases",
            color=color_palette,
        )

        plt2 = Plots.plot(
            df_cum_daily.date,
            [
                df_cum_daily.cum_daily_cases,
                df_cum_daily.cum_daily_hospital_admissions,
                df_cum_daily.cum_daily_deaths,
            ];
            label=["Cumulative cases" "Cumulative hospital_admissions" "Cumulative deaths"],
            title="Cumulative daily data",
            color=color_palette,
        )

        p = Plots.plot(plt1, plt2; layout=(2, 1), size=(800, 600))
        display(p)
        if saveflag
            savefig(p, datadir("raw_data/Fig1_$country.png"))
        end
    end
    return country_data, df_cum_daily, country_population
end

function filter_df_portion(
    df_cum_daily,
    start_from_cum_portion::Float64,
    duration::Int64,
    total_population;
    per_million=false,
)
    start_from_cum_case_ =
        per_million ? start_from_cum_portion : start_from_cum_portion * total_population
    df = df_cum_daily[df_cum_daily.cum_daily_cases .>= start_from_cum_case_, :]
    df = df[1:duration, :]
    return df
end
