document.addEventListener("click", (event) => {
  const link = event.target.closest(
    "#pst-secondary-sidebar-modal #pst-page-toc-nav a[href^='#']",
  );
  if (link) {
    document.getElementById("pst-secondary-sidebar-modal").close();
  }
});
