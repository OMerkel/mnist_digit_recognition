#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive digit drawing and prediction using trained MNIST model."""

import sys
import numpy as np
from tensorflow import keras  # pylint: disable=E0611
import matplotlib.pyplot as plt


class DigitDrawingApp:
    """Interactive application for drawing and predicting digits."""

    def __init__(self, model_path='mnist_model.keras'):
        """Initialize the drawing application."""
        # Load the model
        print(f"Loading model from '{model_path}'...")
        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please run 'uv run train_model.py' first to train and save the model.")
            sys.exit(1)

        # Initialize canvas (28x28 grid)
        self.canvas_size = 28
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)

        # Drawing parameters
        self.is_drawing = False
        self.brush_size = 2  # Brush radius in pixels

        # Setup the figure
        self.setup_figure()

        # Initial prediction
        self.update_prediction()

    def setup_figure(self):
        """Setup the matplotlib figure with canvas and prediction display."""
        self.fig = plt.figure(figsize=(14, 6))

        # Canvas subplot (left)
        self.ax_canvas = plt.subplot(1, 2, 1)
        self.ax_canvas.set_title('Draw a digit (Left click: draw, Right click: clear)',
                                 fontsize=12, weight='bold')
        self.canvas_img = self.ax_canvas.imshow(self.canvas, cmap='gray',
                                                vmin=0, vmax=1, interpolation='nearest')
        self.ax_canvas.set_xlim(-0.5, self.canvas_size - 0.5)
        self.ax_canvas.set_ylim(self.canvas_size - 0.5, -0.5)
        self.ax_canvas.set_xticks([])
        self.ax_canvas.set_yticks([])

        # Add grid for better visibility
        for i in range(self.canvas_size + 1):
            self.ax_canvas.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
            self.ax_canvas.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)

        # Prediction subplot (right)
        self.ax_pred = plt.subplot(1, 2, 2)
        self.ax_pred.set_title('Prediction Probabilities', fontsize=12, weight='bold')
        self.ax_pred.set_xlabel('Probability (%)', fontsize=10)
        self.ax_pred.set_ylabel('Digit', fontsize=10)

        # Initial empty bars
        self.digits = list(range(10))
        self.probs = np.zeros(10)
        self.bars = self.ax_pred.barh(self.digits, self.probs, color='steelblue')
        self.ax_pred.set_xlim(0, 115)  # Extended to accommodate labels
        self.ax_pred.set_ylim(-1.8, 9.5)  # Extended top to accommodate prediction text
        self.ax_pred.set_yticks(self.digits)
        self.ax_pred.invert_yaxis()
        self.ax_pred.grid(axis='x', alpha=0.3)

        # Add percentage labels
        self.prob_labels = []
        for i in range(10):
            label = self.ax_pred.text(
                0, i, '0.00%', va='center', ha='left',
                fontsize=9, weight='bold')
            self.prob_labels.append(label)

        # Predicted digit text - placed inside the axes at the top
        self.pred_text = self.ax_pred.text(
            57.5, -1.2, 'Predicted: -',
            fontsize=14, weight='bold', ha='center',
            transform=self.ax_pred.transData)

        plt.tight_layout()

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.ax_canvas:
            return

        if event.button == 1:  # Left click - start drawing
            self.is_drawing = True
            self.draw_at(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - clear canvas
            self.clear_canvas()

    def on_release(self, event):
        """Handle mouse button release."""
        if event.button == 1:  # Left click released
            self.is_drawing = False

    def on_motion(self, event):
        """Handle mouse motion."""
        if self.is_drawing and event.inaxes == self.ax_canvas:
            self.draw_at(event.xdata, event.ydata)

    def draw_at(self, x, y):
        """Draw on the canvas at the given position."""
        if x is None or y is None:
            return

        # Convert to integer grid coordinates
        cx, cy = int(round(x)), int(round(y))

        # Draw with brush
        for dx in range(-self.brush_size, self.brush_size + 1):
            for dy in range(-self.brush_size, self.brush_size + 1):
                # Check if within circular brush
                if dx*dx + dy*dy <= self.brush_size * self.brush_size:
                    px, py = cx + dx, cy + dy
                    # Check bounds
                    if 0 <= px < self.canvas_size and 0 <= py < self.canvas_size:
                        # Smooth drawing with distance-based intensity
                        dist = np.sqrt(dx*dx + dy*dy)
                        intensity = max(0, 1 - dist / (self.brush_size + 1))
                        self.canvas[py, px] = min(1.0, self.canvas[py, px] + intensity * 0.3)

        # Update display
        self.update_display()

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.fill(0)
        self.update_display()

    def update_display(self):
        """Update the canvas display and prediction."""
        # Update canvas image
        self.canvas_img.set_data(self.canvas)

        # Update prediction
        self.update_prediction()

        # Redraw
        self.fig.canvas.draw_idle()

    def update_prediction(self):
        """Run prediction and update the probability display."""
        # Prepare input for model
        img_input = np.expand_dims(self.canvas, axis=-1)  # Add channel dimension
        img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

        # Predict
        prediction = self.model.predict(img_input, verbose=0)
        probabilities = prediction[0] * 100  # Convert to percentages

        # Get predicted digit
        predicted_digit = np.argmax(probabilities)
        max_prob = probabilities[predicted_digit]

        # Update bars
        for i, (bar, prob) in enumerate(zip(self.bars, probabilities)):
            bar.set_width(prob)
            # Highlight the predicted digit
            if i == predicted_digit:
                bar.set_color('forestgreen')
            else:
                bar.set_color('steelblue')

        # Update percentage labels
        for i, (label, prob) in enumerate(zip(self.prob_labels, probabilities)):
            label.set_text(f'{prob:5.2f}%')
            # Position labels at the end of bars with minimum spacing to avoid overlap
            x_pos = max(prob + 2, 5)  # Minimum x position to prevent clustering
            label.set_position((x_pos, i))
            # Make predicted digit label bold
            if i == predicted_digit:
                label.set_weight('bold')
                label.set_color('forestgreen')
            else:
                label.set_weight('normal')
                label.set_color('black')

        # Update prediction text
        self.pred_text.set_text(f'Predicted: {predicted_digit} ({max_prob:.1f}%)')
        if max_prob > 50:
            self.pred_text.set_color('forestgreen')
        elif max_prob > 20:
            self.pred_text.set_color('orange')
        else:
            self.pred_text.set_color('gray')

    def run(self):
        """Start the interactive application."""
        print("\n" + "=" * 60)
        print("Interactive Digit Drawing & Prediction")
        print("=" * 60)
        print("\nInstructions:")
        print("  • Left click and drag to draw")
        print("  • Right click to clear the canvas")
        print("  • Close the window to exit")
        print("\nDrawing...\n")
        plt.show()


def main():
    """Main entry point."""
    model_path = 'mnist_model.keras'
    app = DigitDrawingApp(model_path)
    app.run()


if __name__ == "__main__":
    main()
