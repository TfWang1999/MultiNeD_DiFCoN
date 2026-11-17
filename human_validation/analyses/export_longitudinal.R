# Export author-normalized RNA data; no normalization/refitting.
# Rscript export_longitudinal.R INPUT_RDS_GZ OUTPUT_DIR REQUIRED_GENES_TXT
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected input RDS, output directory, fixed required-gene list")
library(Matrix)

path <- args[1]
outdir <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

gene_file <- args[3]

required_genes <- readLines(gene_file)
file.copy(gene_file, file.path(outdir, "required_human_genes.txt"), overwrite = TRUE)

cat("Required human genes:", length(required_genes), "\n")



tmp <- tempfile(fileext = ".RDS.gz")
obj <- tryCatch({
    cat("Decompressing first gzip layer...\n")
    src <- gzfile(path, "rb")
    dst <- file(tmp, "wb")
    tryCatch({
        repeat {
            buf <- readBin(src, what = "raw", n = 1024 * 1024)
            if (length(buf) == 0) break
            writeBin(buf, dst)
        }
    }, finally = {
        close(src)
        close(dst)
    })

    cat("Reading inner RDS...\n")
    con <- gzfile(tmp, "rb")
    tryCatch(readRDS(con), finally = close(con))
}, finally = unlink(tmp))

cat("RDS loaded.\n")



meta <- attr(obj, "meta.data", exact = TRUE)



assays <- attr(obj, "assays", exact = TRUE)
rna <- assays[["RNA"]]

expr <- attr(rna, "data", exact = TRUE)

if (is.null(expr)) {
    layers <- attr(rna, "layers", exact = TRUE)

    if (!is.null(layers) && "data" %in% names(layers)) {
        expr <- layers[["data"]]
    }
}

if (is.null(expr)) {
    stop("Normalized RNA expression matrix was not found.")
}

cat(
    "Full RNA matrix:",
    nrow(expr), "genes x",
    ncol(expr), "cells\n"
)



keep <- intersect(required_genes, rownames(expr))

cat("Requested genes:", length(required_genes), "\n")
cat("Found in RNA matrix:", length(keep), "\n")

if (length(keep) == 0) {
    stop("No requested genes found in RNA matrix.")
}

expr_sub <- expr[keep, , drop = FALSE]

cat(
    "Selected matrix:",
    nrow(expr_sub), "genes x",
    ncol(expr_sub), "cells\n"
)



barcodes <- colnames(expr_sub)
meta_sub <- meta[barcodes, , drop = FALSE]

cat(
    "Cell order identical:",
    identical(barcodes, rownames(meta_sub)),
    "\n"
)



cat("Writing sparse matrix...\n")

stopifnot(identical(barcodes, rownames(meta_sub)))

Matrix::writeMM(
    as(expr_sub, "dgCMatrix"),
    file.path(outdir, "program_expression.mtx")
)

writeLines(
    rownames(expr_sub),
    file.path(outdir, "program_genes.txt")
)

writeLines(
    barcodes,
    file.path(outdir, "barcodes.txt")
)

meta_con <- gzfile(
    file.path(outdir, "metadata.csv.gz"),
    "wt"
)

write.csv(
    meta_sub,
    meta_con,
    row.names = TRUE
)

close(meta_con)



rm(expr_sub, expr, obj)
gc()

cat("\nExport finished.\n")
