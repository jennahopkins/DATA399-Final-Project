

library(readr)
#' Reading in Dataset

setwd("C:/Users/jerri/OneDrive/Documents/DATA399/DATA399-Final-Project/Final Project")
ds_fire_og = read_csv("combined_dataset.csv")

# Libraries
library(cluster)
library(tidyverse)
library(FactoMineR)
library(factoextra)
library(umap)
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

# fire_groups_nums = cutree(clusters, k = 3)
# fire_groupings = data.frame(Fire = row.names(fire_data), Group = fire_groups)
# fire_groupings

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
    tmax,
    #vpdmax,
    #canopy_cover,
    #slope_deg,
    #aspect_deg,
    elevation_m,
    fuel_model_group,
    vegetation_type_group) %>%
  mutate(across(where(is.character), as.factor))

dist_matrix = daisy(env_data, metric = "gower")

nmds_result_3d = metaMDS(dist_matrix, k = 3, trymax = 10, autotransform = FALSE)

nmds_matrix_3d = as.matrix(nmds_result_3d$points)

fit = envfit(nmds_result_3d, fire[, c("severity_class", "fire_group",
                                      "vpdmax", "elevation_m", 
                                      "fuel_model_group", 
                                      "vegetation_type_group")], 
              perm = 999)

print(fit)

# ============================================================
# NMDS Pairwise Panels -- Full Clean Pipeline
# ============================================================

# Step 1: Extract site scores from the 3D NMDS result
site_scores <- as.data.frame(nmds_result_3d$points)
colnames(site_scores) <- c("NMDS1", "NMDS2", "NMDS3")
site_scores$fire_group     <- fire$fire_group
site_scores$severity_class <- fire$severity_class

# Step 2: Rerun envfit on the raw points matrix for vectors and factors
fit_vec <- envfit(
  nmds_result_3d$points,
  fire %>% select(vpdmax, elevation_m),
  perm = 999
)

fit_fac <- envfit(
  nmds_result_3d$points,
  fire %>% select(fuel_model_group, vegetation_type_group,
                  severity_class, fire_group),
  perm = 999
)

# Step 3: Build vec_coords and fact_coords safely
vec_coords <- as.data.frame(fit_vec$vectors$arrows)
colnames(vec_coords) <- c("NMDS1", "NMDS2")
vec_coords$NMDS3    <- 0
vec_coords$variable <- rownames(vec_coords)

fact_coords <- as.data.frame(fit_fac$factors$centroids)
colnames(fact_coords) <- c("NMDS1", "NMDS2")
fact_coords$NMDS3    <- 0
fact_coords$variable <- rownames(fact_coords)

# Step 4: Clean up factor labels
#' Keeping only fuel model centroids -- vegetation type adds too many 
#' overlapping labels. Stripping the prefix so "Grass" shows instead 
#' of "fuel_model_groupGrass"

fact_coords_plot <- fact_coords %>%
  filter(grepl("fuel_model_group", variable)) %>%
  mutate(variable = gsub("fuel_model_group", "", variable))

# Step 5: Shared styling
group_colors <- c(
  "Group 1" = "#B85042",
  "Group 2" = "#4A7C59",
  "Group 3" = "#4A6FA5"
)

severity_shapes <- c(
  "High Severity"        = 17,
  "Increased_Vegetation" = 15,
  "Low_Severity"         = 16,
  "Moderate_Severity"    = 18
)

nmds_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 14, hjust = 0.5),
    panel.grid.minor   = element_blank(),
    panel.grid.major   = element_line(color = "gray92"),
    panel.border       = element_rect(color = "gray80", fill = NA),
    plot.background    = element_rect(fill = "white", color = NA),
    axis.title         = element_text(size = 11),
    legend.title       = element_text(face = "bold", size = 14),
    legend.text        = element_text(size = 12),
    plot.margin        = unit(c(10, 15, 10, 15), "pt"),
    legend.key.size    = unit(1.2, "lines")  
  )

# Step 6: Panel builder function
make_nmds_panel <- function(xvar, yvar) {
  
  pts <- site_scores %>% 
    rename(x_axis = all_of(xvar), y_axis = all_of(yvar))
  
  vec <- vec_coords %>% 
    rename(x_axis = all_of(xvar), y_axis = all_of(yvar))
  
  fct <- fact_coords_plot %>% 
    rename(x_axis = all_of(xvar), y_axis = all_of(yvar))
  
  # Scale arrows relative to the data range so they're always visible
  x_range <- diff(range(pts$x_axis, na.rm = TRUE))
  y_range <- diff(range(pts$y_axis, na.rm = TRUE))
  arrow_scale <- min(x_range, y_range) * 0.35
  
  vec <- vec %>%
    mutate(x_end = x_axis * arrow_scale,
           y_end = y_axis * arrow_scale)
  
  ggplot(pts, aes(x = x_axis, y = y_axis,
                  color = fire_group,
                  shape = severity_class)) +
    geom_point(alpha = 0.5, size = 1.5) +
    
    # Environmental vector arrows
    geom_segment(data = vec,
                 aes(x = 0, y = 0, xend = x_end, yend = y_end),
                 inherit.aes = FALSE,
                 arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
                 color = "black", linewidth = 0.9) +
    geom_label(data = vec,
               aes(x = x_end * 1.25, y = y_end * 1.25, label = variable),
               inherit.aes = FALSE,
               size = 5, fontface = "bold", color = "black",
               fill = "white", label.size = 0,
               label.padding = unit(0.15, "lines")) +
    
    # Fuel model centroids with repelled labels
    ggrepel::geom_text_repel(
      data = fct,
      aes(x = x_axis, y = y_axis, label = variable),
      inherit.aes = FALSE,
      color = "red", size = 5, fontface = "bold",
      max.overlaps = 20,
      box.padding = 0.4,
      segment.color = "red", segment.size = 0.3, segment.alpha = 0.5
    ) +
    geom_point(data = fct,
               aes(x = x_axis, y = y_axis),
               inherit.aes = FALSE,
               color = "red", size = 5, shape = 3, stroke = 1.2) +
    
    scale_color_manual(values = group_colors, name = "Fire Group") +
    scale_shape_manual(values = severity_shapes, name = "Severity Class") +
    labs(title = paste(xvar, "vs", yvar), x = xvar, y = yvar) +
    nmds_theme
}

# Step 7: Build and view each panel individually
p1 <- make_nmds_panel("NMDS1", "NMDS2")
p2 <- make_nmds_panel("NMDS1", "NMDS3")
p3 <- make_nmds_panel("NMDS2", "NMDS3")

# View them one at a time in RStudio
p1
p2
p3
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

# Treatment Effect on Severity
#' Testing whether fuel treatment (treated vs untreated) has a statistically
#' significant effect on severity within each fire.
#' Wilcoxon rank-sum test is used here (non-parametric) since we already 
#' established that severity is not normally distributed.

treatment_results <- ds_fire_og %>%
  group_by(fire) %>%
  group_map(~ {
    treated_vals   <- .x$severity[.x$treated == 1]
    untreated_vals <- .x$severity[.x$treated == 0]
    
    # Need at least 3 observations in each group to run the test
    if (length(treated_vals) < 3 | length(untreated_vals) < 3) return(NULL)
    
    test <- wilcox.test(treated_vals, untreated_vals)
    
    tibble(
      fire       = unique(.x$fire),
      median_treated   = median(treated_vals, na.rm = TRUE),
      median_untreated = median(untreated_vals, na.rm = TRUE),
      w_stat     = test$statistic,
      p_value    = test$p.value,
      significant = test$p.value < 0.05
    )
  }, .keep = TRUE) %>%
  bind_rows()

print(treatment_results)

#' Adding treatment as a fixed effect in the mixed effects model.
#' This lets us ask: after accounting for fire identity and the FAMD 
#' dimensions, does treatment status still predict severity?

model_treatment <- lmer(
  severity ~ Dim.1 + Dim.2 + Dim.3 + treated + (1 | fire),
  data = famd_results %>% 
    left_join(ds_fire %>% select(fire, treated) %>% distinct(), by = "fire")
)

summary(model_treatment)

#' Compare to the baseline random intercept model to see if adding
#' treatment improves model fit.
AIC(model_ri, model_treatment)


#' The previous join failed because treated varies pixel-by-pixel, not 
#' by fire. We need to carry treated through the FAMD pipeline instead
#' of joining it back afterward.
#' 
#' The fix is to re-run run_famd() with treated retained in the output,
#' since it was available in the original data at the pixel level.

run_famd_v2 <- function(data) {
  
  feature_cols <- c("ppt", "tmax", "vpdmax", "canopy_cover", 
                    "elevation_m", "fuel_model_group", "vegetation_type_group")
  
  famd_data <- data %>%
    select(all_of(feature_cols), treated, severity) %>%
    drop_na()
  
  if (nrow(famd_data) < 20) return(NULL)
  
  # FAMD runs on features only, not treated or severity
  famd_res <- FactoMineR::FAMD(
    famd_data %>% select(all_of(feature_cols)), 
    graph = FALSE
  )
  
  coords <- as.data.frame(famd_res$ind$coord)
  km <- kmeans(coords, centers = 3, nstart = 25)
  coords$cluster  <- factor(km$cluster)
  coords$fire     <- unique(data$fire)
  coords$severity <- famd_data$severity
  coords$treated  <- famd_data$treated   # ← carried through cleanly
  
  var_contrib <- as.data.frame(famd_res$var$contrib)
  var_contrib$variable <- rownames(var_contrib)
  var_contrib$fire <- unique(data$fire)
  
  list(coords = coords, loadings = var_contrib)
}

results_v2    <- map(fire_list, run_famd_v2)
famd_results  <- map_dfr(results_v2, "coords")
famd_loadings <- map_dfr(results_v2, "loadings")

# Now the mixed effects model will work correctly
model_treatment <- lmer(
  severity ~ Dim.1 + Dim.2 + Dim.3 + treated + (1 | fire),
  data = famd_results
)

summary(model_treatment)
AIC(model_ri, model_treatment)


# PERMANOVA Cluster Validation
#' The PERMANOVA (Permutational Multivariate ANOVA) tests whether the 
#' HDBSCAN clusters correspond to genuinely different multivariate 
#' environmental profiles using the Gower distance matrix.
#' This is a stronger validation than just visual separation in UMAP space --
#' it confirms the clusters differ across ALL input variables simultaneously.
#' 
#' Note: Noise points (cluster == 0) are excluded since HDBSCAN labels 
#' them as unassigned rather than a meaningful group.

permanova_results <- purrr::map_dfr(results, function(res) {
  
  fire_name <- unique(res$coords$fire)
  
  cd <- res$cluster_data %>%
    filter(cluster != 0) %>%     # Remove HDBSCAN noise points
    drop_na(tmax, canopy_cover, elevation_m, 
            fuel_model_group, vegetation_type_group)
  
  if (n_distinct(cd$cluster) < 2 | nrow(cd) < 10) return(NULL)
  
  features <- cd %>%
    select(tmax, canopy_cover, elevation_m,
           fuel_model_group, vegetation_type_group)
  
  gd <- cluster::daisy(features, metric = "gower")
  
  perm <- vegan::adonis2(gd ~ cluster, data = cd, permutations = 999)
  
  tibble(
    fire      = fire_name,
    R2        = perm$R2[1],        # Proportion of variance explained by cluster
    p_value   = perm$`Pr(>F)`[1],
    significant = perm$`Pr(>F)`[1] < 0.05
  )
})

print(permanova_results)

#' R2 here tells you the proportion of multivariate environmental variance
#' explained by cluster membership -- higher means the clusters correspond 
#' to more distinct environmental conditions.
#' The variation in R² across fires is itself a finding — fires with higher R² 
#' have more environmentally stratified landscapes, while Dixie and Camp burned 
#' across more homogeneous terrain, suggesting severity there was driven more by 
#' fire behavior dynamics (wind, spread rate) than by the static environmental features 
#' we measured.


# NMDS envfit Variable Importance Summary
#' Extracting R² and p-values from the envfit object to create a ranked 
#' summary of which environmental variables most strongly align with 
#' the ordination space -- and therefore with severity patterns across fires.
#' 
#' Two separate tables are built: one for continuous vectors (tmax, vpdmax, 
#' elevation) and one for categorical factors (fuel model, vegetation type,
#' severity class, fire group), since envfit handles them differently.

# Continuous variables
vector_r2 <- as.data.frame(fit$vectors$r)
colnames(vector_r2) <- "R2"
vector_r2$p_value <- fit$vectors$pvals
vector_r2$variable <- rownames(vector_r2)
vector_r2$type <- "Continuous"

# Categorical variables
factor_r2 <- as.data.frame(fit$factors$r)
colnames(factor_r2) <- "R2"
factor_r2$p_value <- fit$factors$pvals
factor_r2$variable <- rownames(factor_r2)
factor_r2$type <- "Categorical"

envfit_summary <- bind_rows(vector_r2, factor_r2) %>%
  arrange(desc(R2)) %>%
  mutate(significant = p_value < 0.05)

print(envfit_summary)

# Ranked bar chart of R² values by variable

#' Variables with high R² and significant p-values are the ones most 
#' strongly structuring the ordination -- these are your best candidates
#' for explaining cross-fire severity patterns in the white paper.
#' 
#'fuel characteristics (fuel model and vegetation type) are the primary
#'structuring forces of fire severity patterns across the landscape, explaining
#' roughly 65–75% of ordination variance, while climate stress (VPDmax) and topography
#'(elevation) play a secondary but still meaningful role. Fire identity and severity
#'classification contribute minimally to ordination structure, indicating that
#'environmental features drive the underlying spatial organization of severity.


envfit_summary <- envfit_summary %>%
  mutate(variable_label = case_when(
    variable == "fuel_model_group"      ~ "Fuel Model Type",
    variable == "vegetation_type_group" ~ "Vegetation Type",
    variable == "elevation_m"           ~ "Elevation",
    variable == "vpdmax"                ~ "Atmospheric Dryness (VPD)",
    variable == "severity_class"        ~ "Severity Class",
    variable == "fire_group"            ~ "Fire Group",
    TRUE ~ variable
  ))

ggplot(envfit_summary, 
       aes(x = reorder(variable_label, R2), y = R2)) +
  geom_col(width = 0.65, fill = "#C51616") +
  
  # R² value labels on the end of each bar
  geom_text(aes(label = round(R2, 2)),
            hjust = -0.15,
            fontface = "bold",
            size = 5,
            color = "gray20") +
  
  coord_flip(clip = "off") +
  expand_limits(y = 0.9) +   # gives room for the labels past the bars
  
  labs(
    title    = "What Drives Wildfire Severity Patterns?",
    x        = NULL,         # y-axis label is self explanatory with readable names
    y        = "R² (Proportion of Variance Explained)"
  ) +
  
  theme_minimal(base_size = 15) +
  theme(
    plot.title       = element_text(face = "bold", size = 20, hjust = 0),
    axis.text.y      = element_text(face = "bold", size = 14, color = "gray10"),
    axis.text.x      = element_text(size = 12, color = "gray10"),
    axis.title.x     = element_text(size = 13, color = "gray10",
    panel.grid.major.y = element_blank(),   # removes horizontal gridlines
    panel.grid.major.x = element_line(color = "gray90"),
    legend.position  = "none",
    plot.background  = element_rect(fill = "white", color = NA))
  )
