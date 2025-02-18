#!/usr/bin/env python3
"""
user_input.py - A script to create presentations from user input using Streamlit.
Requires: streamlit, typing
"""

import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path


@dataclass
class Slide:
    """Represents a single presentation slide."""
    title: str
    content: str
    image_path: Optional[str] = None


class PresentationManager:
    """Manages the creation and display of presentations."""

    def __init__(self) -> None:
        """Initialize the presentation manager."""
        self.slides: List[Slide] = []

    def add_slide(self, slide: Slide) -> None:
        """
        Add a new slide to the presentation.
        
        Args:
            slide: Slide object containing title and content
        """
        try:
            self.slides.append(slide)
        except Exception as e:
            st.error(f"Error adding slide: {str(e)}")

    def get_user_input(self) -> None:
        """Collect user input for presentation slides."""
        try:
            st.title("Create Your Presentation")
            
            with st.form("new_slide"):
                slide_title = st.text_input("Slide Title")
                slide_content = st.text_area("Slide Content")
                image_file = st.file_uploader("Upload Image (optional)", 
                                            type=['png', 'jpg', 'jpeg'])
                
                if st.form_submit_button("Add Slide"):
                    if slide_title and slide_content:
                        image_path = None
                        if image_file:
                            # Save uploaded image
                            save_path = Path(f"uploads/{image_file.name}")
                            save_path.parent.mkdir(exist_ok=True)
                            save_path.write_bytes(image_file.getvalue())
                            image_path = str(save_path)
                        
                        new_slide = Slide(
                            title=slide_title,
                            content=slide_content,
                            image_path=image_path
                        )
                        self.add_slide(new_slide)
                        st.success("Slide added successfully!")
                    else:
                        st.warning("Please fill in both title and content.")

    def display_presentation(self) -> None:
        """Display the created presentation."""
        try:
            if not self.slides:
                st.info("No slides to display. Please add some slides first.")
                return

            st.title("Your Presentation")
            
            current_slide = st.session_state.get('current_slide', 0)
            
            # Navigation buttons
            cols = st.columns(3)
            with cols[0]:
                if st.button("Previous") and current_slide > 0:
                    current_slide -= 1
            with cols[2]:
                if st.button("Next") and current_slide < len(self.slides) - 1:
                    current_slide += 1
            
            # Display current slide
            slide = self.slides[current_slide]
            st.header(slide.title)
            st.write(slide.content)
            
            if slide.image_path:
                try:
                    st.image(slide.image_path, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            
            # Update session state
            st.session_state.current_slide = current_slide
            
            # Display slide counter
            st.write(f"Slide {current_slide + 1} of {len(self.slides)}")

        except Exception as e:
            st.error(f"Error displaying presentation: {str(e)}")


def main() -> None:
    """Main function to run the presentation application."""
    try:
        st.set_page_config(page_title="Presentation Creator",
                          layout="wide")
        
        presentation = PresentationManager()
        
        # Sidebar navigation
        page = st.sidebar.radio("Navigation",
                               ["Create Slides", "View Presentation"])
        
        if page == "Create Slides":
            presentation.get_user_input()
        else:
            presentation.display_presentation()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()