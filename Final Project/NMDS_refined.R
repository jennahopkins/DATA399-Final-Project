

sig_vec_idx <- which(fit$vectors$pvals < 0.05)
vec_coords_sig <- as.data.frame(fit$vectors$arrows[sig_vec_idx, , drop = FALSE])
vec_coords_sig$variable <- rownames(vec_coords_sig)

# --- Filter Factors (Categorical like fuel_model) ---
sig_fact_idx <- which(fit$factors$pvals < 0.05)
# This identifies which CATEGORIES (e.g., 'Timber') belong to significant groups
fact_coords_sig <- as.data.frame(fit$factors$centroids[sig_fact_idx, , drop = FALSE])
fact_coords_sig$variable <- rownames(fact_coords_sig)


plot_3d_clean <- plot_ly() %>%
  # 1. Add Data Points (Clean Legend)
  add_trace(data = site_scores, 
            x = ~NMDS1, y = ~NMDS2, z = ~NMDS3,
            color = ~fire_group,          # One color per group
            symbol = ~severity_class,     # Different shapes for severity
            type = "scatter3d", 
            mode = "markers",
            marker = list(size = 3, opacity = 0.6),
            # This 'text' appears when you hover over a point
            text = ~paste("Fire Group:", fire_group, 
                          "<br>Severity:", severity_class),
            hoverinfo = "text") %>%
  
  layout(scene = list(aspectmode = "cube",
                      xaxis = list(title = "NMDS1"),
                      yaxis = list(title = "NMDS2"),
                      zaxis = list(title = "NMDS3")),
         legend = list(title = list(text = '<b>Fire Groups</b>')))

plot_3d_clean


for (i in 1:nrow(vec_coords_sig)) {
  plot_3d_clean <- plot_3d_clean %>%
    add_trace(x = c(0, vec_coords_sig$NMDS1[i]), 
              y = c(0, vec_coords_sig$NMDS2[i]), 
              z = c(0, vec_coords_sig$NMDS3[i]),
              type = "scatter3d", mode = "lines+text",
              line = list(color = "black", width = 5),
              text = c("", vec_coords_sig$variable[i]), # Label only at the tip
              textposition = "top center",
              name = paste("Driver:", vec_coords_sig$variable[i]),
              showlegend = TRUE)
}

# 3. Add Significant Factors (Red Text Centroids)
plot_3d_clean <- plot_3d_clean %>%
  add_trace(data = fact_coords_sig, 
            x = ~NMDS1, y = ~NMDS2,
            type = "scatter3d", mode = "text",
            text = ~variable, 
            textfont = list(color = "red", size = 11),
            name = "Veg/Fuel Types",
            showlegend = TRUE)

plot_3d_clean


