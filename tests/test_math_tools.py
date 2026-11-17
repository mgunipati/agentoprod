import unittest
from unittest.mock import MagicMock

from strands.models import Model

from app.agendev.math_tools import create_math_tools, create_tutor_agents


class MathToolsTests(unittest.TestCase):
    def test_tutors_share_the_supplied_model(self):
        model = MagicMock(spec=Model)

        tutors = create_tutor_agents(model)

        self.assertEqual(len(tutors), 3)
        self.assertTrue(all(tutor.model is model for tutor in tutors))

    def test_math_tools_are_wrapped_strands_agent_tools(self):
        tools = create_math_tools(MagicMock(spec=Model))

        self.assertEqual(
            [tool.tool_name for tool in tools],
            ["math_tutor", "writing_coach", "science_helper"],
        )


if __name__ == "__main__":
    unittest.main()
