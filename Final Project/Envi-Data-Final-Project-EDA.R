


#' Reading in Dataset

setwd("/Users/anujantennathur/Documents/GitHub/EnvDataSci_Sp26/Final Project")
ds_fire_og = read_csv("combined_dataset.csv")

# Libraries

library(tidyverse)
library(FactoMineR)
library(factoextra)
library(umap)
library(cluster)
library(GGally)
library(plotly)
library(dbscan)
library(patchwork)
library(purrr)
library(randomForest)
library(reshape2)
library(lme4)
library(lawstat)
library(rstatix)
library(vegan)
library(nortest)
library(FSA)
library(vegan)
library(scatterplot3d)


# Base Visualizations, EDA

# Histograms

ds_fire_og %>%
  ggplot(aes(x = severity)) +
  geom_histogram(aes(y = after_stat(density)), 
                 binwidth = 25,
                 color = "cyan",
                 fill = "black") +
  # geom_density(color = "red", size = 1) +
  coord_cartesian(xlim = c(-1000, 1300)) +
  labs(
    title = "Distribution of Severity -- Excludes Outliers",
    x = "Severity (Relative Differenced Normalized Burned Ratio)",
    y = ""
  ) +
  facet_wrap(~fire) +
  theme_minimal()


ds_fire_og %>%
  ggplot(aes(x = canopy_cover)) +
  geom_bar()+
  # geom_density(color = "red", size = 1) +
  coord_cartesian(xlim = c(15, 60)) +
  labs(
    title = "Distribution of Canopy Cover",
    x = "Canopy Cover (in Percentage)",
    y = ""
  ) + 
  facet_wrap(~fire) + 
  theme_minimal()



Antelope_2021 = ds_fire_og %>%
  filter(fire == "Antelope_2021")

Bald_2014 = ds_fire_og %>%
  filter(fire == "Bald_2014")

Carr_2018 = ds_fire_og %>%
  filter(fire == "Carr_2018")

McKinney_2022 = ds_fire_og %>%
  filter(fire == "McKinney_2022")

Dixie_2021 = ds_fire_og %>%
  filter(fire == "Dixie_2021")

Monument_2021 = ds_fire_og %>%
  filter(fire == "Monument_2021")

King_2014 = ds_fire_og %>%
  filter(fire == "King_2014")

NorthComplex_2020 = ds_fire_og %>%
  filter(fire == "NorthComplex_2020")

Hat_2018 = ds_fire_og %>%
  filter(fire == "Hat_2018")

Camp_2018 = ds_fire_og %>%
  filter(fire == "Camp_2018")

fire_list = list(
  Antelope_2021, Bald_2014, Carr_2018, McKinney_2022, Dixie_2021, 
  Monument_2021, King_2014, NorthComplex_2020, Hat_2018, Camp_2018
)

# Testing for Normality, Distribution Type of Severities
# Performed on each fire

#' Two tests were performed, Shapiro test and Anderson Darling test.
#' Note: Anderson Darling test is even more sensitive to outlier 
#' data compared to Shapiro test. 
#' Both tests indicate the data is not normally distributed
#' (low p-values and A statistic). 
#' We will proceed with a Kruskal-Wallis test instead to compare 
#' medians across fire groups.

shapiro_results = lapply(fire_list, function(df) shapiro.test(df$severity))


names(shapiro_results) <- c("Antelope", "Bald", "Carr", "McKinney", "Dixie", 
                            "Monument", "King", "NorthComplex", "Hat", "Camp")

print(shapiro_results) 

ad_results = lapply(fire_list, function(df) ad.test(df$severity))

names(ad_results) = names(shapiro_results)

print(ad_results)


#' Kruskal-Wallis Test for Severity Variable

kruskal.test(severity ~ fire, data = ds_fire_og)

#Kruskal-Wallis rank sum test
#data:  severity by fire
#Kruskal-Wallis chi-squared = 90.82, df = 9, p-value = 1.115e-15

#' Interpretation: 
#' The result indicates that there is a statistically significant difference
#' in severity in AT LEAST two of the fires. 
#' - significant p value (<0.05), reject null hypothesis, meaning that median 
#' severity is different in at least one fire compared to others. 
#' - large chi-squared value (90.82). Indicates that the "ranks" of 
#' severity values are not distributed evenly across the fire groups.

#' The Kruskal-Wallis test provides a result that proves differences exist,
#' but cannot provide which fires are differing from each other or not.
#' This is the reason for Dunn's test, which will do pairwise comparisons.

# Dunn's Test Pairwise Comparisons

dunn_results = dunnTest(severity ~ fire, 
                         data = ds_fire_og, 
                         method = "holm")

print(dunn_results)

summary_table = dunn_results$res[, c("Comparison", "P.adj")]
#summary_table

# Statistically Significant Different Fires
significant_fires = subset(dunn_results$res, P.adj < 0.05)
significant_fires = significant_fires[order(significant_fires$P.adj), ]
print(significant_fires[, c("Comparison", "P.adj")])

# Statistically Insignificant Different Fires (Meaning more similarity)
similar_fires = subset(dunn_results$res, P.adj > 0.05)
similar_fires = similar_fires[order(similar_fires$P.adj), ]
print(similar_fires[, c("Comparison", "P.adj")])

#' Through our Dunn's test, we have established:
#' There are 12 combinations (pairs) of fires that are statistically 
#' significantly different in their severity distributions (p<0.05),
#' and there are 33 combinations (pairs) of fires that are NOT statistically
#' significantly different in their severity distributions (p>0.05).
#' These results can be found running the above code.


#' Now that we have established that there are statistically significant 
#' differences between fire severities (Kruskall-Wallis), and we also know 
#' from the Dunn's test which combinations/pairs of fires are either 
#' statistically significantly different, or NOT statistically significantly 
#' different. 
#' 
#' With the above established, let's use Hierarchical Cluster Analysis 
#' abbreviated HCA, an unsupervised machine learning method that builds a 
#' hierarchy of clusters by calculating Euclidean distances in, and presents
#' the results in the form of a dendrogram. 

# HCA Plot

fire_medians = aggregate(severity ~ fire, data = ds_fire_og, FUN = median)
dist_matrix = dist(fire_medians$severity)
clusters = hclust(dist_matrix)
plot(clusters, labels = fire_medians$fire, main = "Fire Similarity Groups")

#' In the above HCA plot, we can see the groupings that the unsupervised 
#' technique suggests, but it is not as precise as we would like to be
#' going forward. We need to know:
#' - the exact number of clusters
#' - what fires should go in which cluster 
#' Thus, the HCA plot useful going forward, but needs to be refined.

#' In order to refine it, we will calculate the 
#' "Total Within-Cluster Sum of Squares" for different numbers of clusters (k).
#' In this case, I tested from 1-9 clusters in the code below.

# Total Within-Cluster Sum of Squares Calculation

row.names(fire_medians) = fire_medians$fire
vals = as.matrix(fire_medians$severity)

wss = sapply(1:9, function(k) {
  group = cutree(clusters, k = k)
  sum(sapply(unique(group), function(g) {
    sum((vals[group == g] - mean(vals[group == g]))^2)
  }))
})

#' Once we have our statistic calculated for different cluster numbers, 
#' we create a plot with the number of clusters on the x-axis and 
#' the aforementioned statistic as the y-axis. By doing this, we are 
#' examining the number of clusters that are effectively reducing the 
#' variance in the statistic.

plot(1:9, wss, type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main="Elbow Method for Fire Groups")

#' With the plot, we will use the Elbow method, which is essentially how 
#' we pick the number of clusters, or groups, that the fires should go in.
#' In the case of this plot, the Elbow method suggests that 3 clusters 
#' is the ideal number, as at 3 clusters, we have reduced the statistic 
#' significantly, and having 4 or more clusters will lead to unnecessary 
#' over-fitting, where we are splitting groups that are already statistically
#' similar. 

fire_groups_nums = cutree(clusters, k = 3)
fire_groupings = data.frame(Fire = row.names(fire_data), Group = fire_groups)
fire_groupings

# Now, we have the groups for the fires as listed below 
# (both names and datasets).

group_1_names = c("Antelope_2021", "Camp_2018", "Hat_2018", "King_2014", 
                  "McKinney_2022")
group_2_names = c("Bald_2014", "Carr_2018")
group_3_names = c("Dixie_2021", "Monument_2021", "NorthComplex_2020")

group_1_fires = c(Antelope_2021, Camp_2018, Hat_2018, King_2014, McKinney_2022)
group_2_fires = c(Bald_2014, Carr_2018)
group_3_fires = c(Dixie_2021, Monument_2021, NorthComplex_2020)

# We are now able to proceed with the rest of the analysis. 

#' NMDS - Non-Metric Multidimensional Scaling - Non-linear
#' 
#' Description (AI used, reword for white paper):
#' 
#' nonlinear dimensionality reduction technique used in machine learning and 
#' multivariate statistics to visualize similarity or dissimilarity between 
#' complex datasets. NMDS focuses on preserving the rank-order of distances 
#' between samples, making it ideal for ecological, microbial, and 
#' high-dimensional data that do not follow linear distributions.
#' NMDS is commonly used as a tool for visualization and exploratory 
#' data analysis (EDA) rather than direct prediction. 
#' It helps identify clusters, patterns, and gradients in data.
#' 
#' Another important method used in this ML pipeline is Gower's distance metric. 
#' The reason that Gower's distance is required for this scenario is because
#' of the type of explanatory data. The Bray-Curtis distance, which is used
#' to calculate the distance for the NMDS ML method, requires a 
#' numerical matrix. Since the Gower's distance method effectively handles
#' both numerical and categorical data and creates distance between points, 
#' it is used on our data prior to feeding the data into the NMDS function.

# Data Wrangling/Cleaning Prior to ML Model

fire = ds_fire_og %>%
  select(-x, -y) %>%
  mutate(
    fire = factor(fire),
    treated = factor(treated),
    fuel_model = factor(fuel_model),
    vegetation_type = factor(vegetation_type),
    fuel_model_group = factor(fuel_model_group),
    vegetation_type_group = factor(vegetation_type_group)
  )

fire = fire %>%
  group_by(fire) %>%
  filter(
    severity > quantile(severity, 0.01, na.rm=TRUE),
    severity < quantile(severity, 0.99, na.rm=TRUE)
  ) %>%
  ungroup()


fire = fire %>%
  mutate(
    severity_class = case_when(
      severity <= -100 ~ "Increased_Vegetation",
      severity >= -100 & severity <= 100 ~ "Low_Severity",
      severity >= 100 & severity <= 270 ~ "Moderate_Severity",
      TRUE ~ "High Severity"
    ),
    severity_class = factor(severity_class)
  )

fire = fire %>%
  mutate(
    fire_group = case_when(
      fire %in% group_1_names ~ "Group 1",
      fire %in% group_2_names ~ "Group 2",
      fire %in% group_3_names ~ "Group 3",
    ),
    fire_group = factor(fire_group)
  )


# NMDS Plot Code

fire = fire %>%
  filter(!is.na(severity_class), 
         !is.na(fire_group),
         !is.na(vpdmax),
         !is.na(elevation_m),
         !is.na(fuel_model_group),
         !is.na(vegetation_type_group))

env_data = fire %>%
  select(
    #ppt,
    #tmax,
    vpdmax,
    #canopy_cover,
    #slope_deg,
    #aspect_deg,
    elevation_m,
    fuel_model_group,
    vegetation_type_group) %>%
  mutate(across(where(is.character), as.factor))

dist_matrix = daisy(env_data, metric = "gower")

nmds_result_3d = metaMDS(dist_matrix, k = 3, trymax = 25, autotransform = FALSE)

nmds_matrix_3d = as.matrix(nmds_result_3d$points)

fit = envfit(nmds_scores_3d, fire[, c("severity_class", "fire_group",
                                      "vpdmax", "elevation_m", 
                                      "fuel_model_group", 
                                      "vegetation_type_group")], 
              perm = 999)

print(fit)

#' Since we have both continuous and categorical factors in our
#' fit variable, we need to extract them before we create 
#' the NMDS Visualization.

site_scores = as.data.frame(scores(nmds_result_3d, display = "sites"))
site_scores$fire_group = fire$fire_group
site_scores$severity_class = fire$severity_class
vec_coords = as.data.frame(scores(fit, display = "vectors"))
vec_coords$variable = rownames(vec_coords)
fact_coords = as.data.frame(scores(fit, display = "factors"))
fact_coords$variable = rownames(fact_coords)

# NMDS Visualization

plot_3d <- plot_ly() %>%
  add_trace(data = site_scores, 
            x = ~NMDS1, y = ~NMDS2, z = ~NMDS3,
            color = ~fire_group, 
            symbol = ~severity_class,
            type = "scatter3d", 
            mode = "markers",
            marker = list(size = 3, opacity = 0.7))

plot_3d


for (i in 1:nrow(vec_coords)) {
  plot_3d = plot_3d %>%
    add_trace(x = c(0, vec_coords$NMDS1[i]), 
              y = c(0, vec_coords$NMDS2[i]), 
              z = c(0, vec_coords$NMDS3[i]),
              type = "scatter3d", mode = "lines",
              line = list(color = "black", width = 4),
              name = vec_coords$variable[i],
              showlegend = TRUE)
}

plot_3d

plot_3d = plot_3d %>%
  add_trace(data = fact_coords, x = ~NMDS1, y = ~NMDS2, 
            # z = ~NMDS3,
            type = "scatter3d", mode = "text",
            text = ~variable, textfont = list(color = "red", size = 12),
            name = "Factors/Vegetation")

plot_3d









 







#' Possible Unsupervised Models
#' -- Use the unsupervised machine learning models to explain variance, not to 
#' explain causal relationships.
#' 
#' 

#' K-Means Clustering, FAMD, PCA --> See if there is linear relationship between variables
#' when trying to explain the variance in severity, faceted by each fire. Then, 
#' we will compare them. 
#' 
#' Non-Linear Clustering Options: UMAPS, Gower's Distances --> Use these if it is 
#' found that linear relationships do not accurately represent the data and the 
#' relationship between the variables. 
#' 

# Unsupervised, Linear Model Pipeline

ds_fire = ds_fire_og %>%
  select(-x, -y) %>%
  mutate(
    fire = factor(fire),
    treated = factor(treated),
    fuel_model = factor(fuel_model),
    vegetation_type = factor(vegetation_type),
    fuel_model_group = factor(fuel_model_group),
    vegetation_type_group = factor(vegetation_type_group)
  )

ds_fire = ds_fire %>%
  group_by(fire) %>%
  filter(
    severity > quantile(severity, 0.01, na.rm=TRUE),
    severity < quantile(severity, 0.99, na.rm=TRUE)
  ) %>%
  ungroup()


ds_fire = ds_fire %>%
  mutate(
    severity_class = case_when(
      severity <= -100 ~ "Increased_Vegetation",
      severity >= -100 & severity <= 100 ~ "Low_Severity",
      severity >= 100 & severity <= 270 ~ "Moderate_Severity",
      TRUE ~ "High Severity"
    ),
    severity_class = factor(severity_class)
  )

ds_fire = ds_fire %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .)))

num_vars = ds_fire %>%
  select(severity, ppt, tmax, vpdmax, canopy_cover, slope_deg, aspect_deg, elevation_m)

cor_matrix = cor(num_vars, use="pairwise.complete.obs")

corrplot::corrplot(cor_matrix, method="color")
GGally::ggpairs(num_vars)

severity_cor = cor_matrix["severity", ]
severity_cor

fire_list = ds_fire %>%
  group_split(fire)



# FAMD + K-Means

# Single Fire Example


ds_fire <- ds_fire %>%
  filter(!is.na(severity))

run_famd <- function(data) {
  famd_data <- data %>%
    select(ppt, 
           tmax, 
           vpdmax,
           canopy_cover, 
           elevation_m,
           fuel_model_group, 
           vegetation_type_group) %>%
    drop_na()
  
  if (nrow(famd_data) < 20) return(NULL)
  
  famd_res <- FactoMineR::FAMD(famd_data, graph = FALSE)
  
  coords <- as.data.frame(famd_res$ind$coord)
  km <- kmeans(coords, centers = 3, nstart = 25)
  coords$cluster <- factor(km$cluster)
  coords$fire <- unique(data$fire)
  
  # ← removed duplicate coords reassignment, just add severity directly
  coords$severity <- data %>% 
    select(severity, ppt, tmax, vpdmax,
           canopy_cover, slope_deg, aspect_deg, elevation_m,
           fuel_model_group, vegetation_type_group, treated) %>%
    drop_na() %>% 
    pull(severity)
  
  var_contrib <- as.data.frame(famd_res$var$contrib)
  var_contrib$variable <- rownames(var_contrib)
  var_contrib$fire <- unique(data$fire)
  
  list(coords = coords, loadings = var_contrib)
}

results <- map(fire_list, run_famd)

famd_results  <- map_dfr(results, "coords")
famd_loadings <- map_dfr(results, "loadings")

# Cluster Plot

ggplot(famd_results, aes(Dim.1, Dim.2, color=cluster)) +
  geom_point(alpha=.6) +
  facet_wrap(~fire) +
  theme_minimal()

# Dimension Variable Weights

famd_loadings %>%
  group_by(variable) %>%
  summarise(Dim.1 = mean(Dim.1), Dim.2 = mean(Dim.2)) %>%
  pivot_longer(cols = c(Dim.1, Dim.2), 
               names_to = "dimension", values_to = "contribution") %>%
  ggplot(aes(x = reorder(variable, contribution), 
             y = contribution, fill = dimension)) +
  geom_col(position = "dodge") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Variable", y = "% Contribution", 
       title = "Average Variable Contributions to FAMD Dimensions")

## Interpretation Models

famd_results %>%
  group_by(fire) %>%
  group_map(~ summary(lm(severity ~ Dim.1 + Dim.2 + Dim.3, data = .x)))

# Get the name of the Fire

famd_results %>%
  distinct(fire) %>%
  arrange(fire) %>%
  slice(6)

fire_models <- famd_results %>%
  group_by(fire) %>%
  group_map(~ {
    fire_name <- unique(.x$fire)
    
    lm_mod <- lm(severity ~ Dim.1 + Dim.2 + Dim.3, data = .x)
    rf_mod <- randomForest(severity ~ Dim.1 + Dim.2 + Dim.3, data = .x, ntree = 500)
    
    tibble(
      fire = fire_name,
      lm_r2 = summary(lm_mod)$r.squared,
      rf_r2 = rf_mod$rsq[500]  # R² at last tree
    )
  }) %>%
  bind_rows()

fire_models

# ----- 

# Mixed Effects Models

library(lme4)

# Random intercept model - fire shifts baseline severity
model_ri <- lmer(severity ~ Dim.1 + Dim.2 + Dim.3 + (1 | fire), 
                 data = famd_results)
summary(model_ri)

# Random intercept + random slopes - fire can also respond differently to each dim
model_rs <- lmer(severity ~ Dim.1 + Dim.2 + Dim.3 + 
                   (1 + Dim.1 + Dim.2 + Dim.3 | fire), 
                 data = famd_results)
summary(model_rs)

# Compare the two models
AIC(model_ri, model_rs)

performance::icc(model_ri)

model_mid <- lmer(severity ~ Dim.1 + Dim.2 + Dim.3 + 
                    (1 + Dim.2 | fire), 
                  data = famd_results)
summary(model_mid)
AIC(model_ri, model_mid, model_rs)





#' Fire [[6]] (King, 2014) stands out — 50% of severity variance explained is 
#' genuinely strong for fire behavior modeling, suggesting predictor 
#' structure maps cleanly onto severity for that fire.
#' Dim.2 is the most consistently significant predictor across fires 
#' (significant in 6 of 9), suggesting whatever that dimension 
#' captures is the most broadly relevant to severity. 
#' 
#' The broader implication is that the predictor-severity relationship is 
#' highly fire-specific, which makes ecological sense — the same topographic 
#' or weather conditions may drive severity very differently depending on 
#' fuel state, ignition pattern, etc. This suggests 
#' a mixed effects model with fire as a random effect might be worth 
#' exploring as an alternative to fitting 9 separate models.


# Mixed Effects Model -- Fire as Random Effect

















# Non-Linear, Gower's Distance UMAPS HBDSCAN

ds_fire <- ds_fire %>%
  filter(!is.na(severity))

run_fire_pipeline <- function(data){
  
  fire_name <- unique(data$fire)
  
  feature_cols <- c(
    #"ppt",
    "tmax",
    #"vpdmax",
    "canopy_cover",
    #"slope_deg",
    #"aspect_deg",
    "elevation_m",
    "fuel_model_group",
    "vegetation_type_group"
  )
  
  features <- data %>%
    select(all_of(feature_cols)) %>%
    drop_na()
  
  n <- nrow(features)
  
  n <- nrow(features)
  
  # Skip fires that are too small
  if(n < 10){
    return(NULL)
  }
  
  # -------------------------
  # GOWER DISTANCE
  # -------------------------
  
  gower_dist <- cluster::daisy(features, metric = "gower")
  
  # -------------------------
  # UMAP (dynamic neighbors)
  # -------------------------
  
  neighbors <- min(10, n - 1)
  
  umap_config <- umap::umap.defaults
  umap_config$n_neighbors <- neighbors
  
  umap_res <- umap::umap(as.matrix(gower_dist), config = umap_config)
  
  coords <- as.data.frame(umap_res$layout)
  colnames(coords) <- c("UMAP1","UMAP2")
  
  # -------------------------
  # HDBSCAN
  # -------------------------
  
  minpts <- max(5, floor(n * 0.05))
  
  hdb <- dbscan::hdbscan(coords, minPts = minpts)
  
  coords$cluster <- factor(hdb$cluster)
  coords$fire <- fire_name
  
  cluster_data <- data %>%
    slice(1:n) %>%
    mutate(cluster = coords$cluster)
  
  list(
    coords = coords,
    cluster_data = cluster_data
  )
}

results <- purrr::map(fire_list, run_fire_pipeline)

results <- results[!sapply(results, is.null)]

umap_results <- purrr::map_dfr(results, "coords")
cluster_data <- purrr::map_dfr(results, "cluster_data")



ggplot(umap_results,
       aes(UMAP1, UMAP2, color = cluster)) +
  geom_point(alpha = .7) +
  facet_wrap(~fire) +
  theme_minimal() +
  labs(title = "UMAP Clusters by Fire")

ggplot(cluster_data,
       aes(cluster, severity, fill = cluster)) +
  geom_boxplot() +
  facet_wrap(~fire) +
  theme_minimal() +
  labs(title = "Severity Distribution by Cluster")

cluster_summary <- cluster_data %>%
  group_by(fire, cluster) %>%
  summarise(
    across(
      c(tmax, canopy_cover,
        elevation_m),
      median,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

cluster_summary

cluster_long <- cluster_summary %>%
  pivot_longer(
    -c(fire, cluster),
    names_to = "variable",
    values_to = "value"
  )

ggplot(cluster_long,
       aes(cluster, variable, fill = value)) +
  geom_tile() +
  facet_wrap(~fire, scales = "free") +
  scale_fill_viridis_c() +
  theme_minimal()

ggplot(cluster_data,
       aes(cluster, fill = fuel_model_group)) +
  geom_bar(position = "fill") +
  facet_wrap(~fire) +
  theme_minimal() +
  labs(
    y = "Proportion",
    title = "Fuel Model Composition by Cluster"
  )


aov(severity ~ cluster, data = cluster_data)
chisq.test(table(cluster_data$fuel_model_group,
                 cluster_data$cluster))







