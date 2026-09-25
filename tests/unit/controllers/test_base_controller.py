from controllers.base_controller import BaseController


def test_generate_random_string():
    controller = BaseController()

    result = controller.generate_random_string(length=12)

    assert len(result) == 12
    assert result.islower()
    assert result.isalnum()
